from __future__ import annotations

import cv2
import os
import torch
import numpy as np
import json
import re
import importlib.util
import time
from collections import OrderedDict, deque, Counter
from PIL import Image
from scipy.optimize import linear_sum_assignment

from transformers import CLIPVisionModel98

try:
    from transformers import Qwen2VLForConditionalGeneration
except Exception:
    Qwen2VLForConditionalGeneration = None

try:
    from transformers import AutoProcessor
except Exception:
    AutoProcessor = None

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None


# ----------------------------
# CONFIG
# ----------------------------
VIDEO_PATH = "10THAU.mp4"
SCENE_JSON_PATH = "semantic_scenes.json"
RESULTS_JSON_PATH = "scene_level_results.json"
SCORES_JSON_PATH = "scene_scores.json"
SELECTION_SUMMARY_PATH = "scene_selection_summary.json"
VLM_CACHE_PATH = "vlm_cache.json"
WHISPERX_JSON_PATH = "whisperx_results.json"
SCRIPT_JSON_PATH = "script_structured.json"
IDENTITY_RECONCILIATION_PATH = "identity_reconciliation.json"
SELECTED_REFERENCE_CLUSTERS_PATHS = [
    "selected_reference_clusters.json",
    os.path.join("report", "selected_reference_clusters.json"),
]

ENABLE_MULTIMODAL_PRIOR = True
ENABLE_IDENTITY_RECONCILIATION = False
TEXT_BOUNDARY_WEIGHT = 0.35
FACE_BOUNDARY_WEIGHT = 0.20
TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPEAKER_CHARACTER_MIN_SIM = 0.08
FACE_CHARACTER_MIN_SIM = 0.12

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

MAX_FRAMES_PER_SCENE = 1
CLIP_BATCH_SIZE = 64
CLIP_USE_HALF = DEVICE == "cuda"
ENABLE_TORCH_COMPILE = True
TORCH_COMPILE_MODE = "reduce-overhead"
MEMORY_SIZE = 50
ENABLE_LLM = True
MULTIMODAL_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
PREFER_FAST_SINGLE_GPU = True
FALLBACK_TO_AUTO_ON_OOM = True
ENABLE_4BIT_FALLBACK = True

# Limits to avoid overloading VLM stage
VLM_MAX_SCENES = 60  # maximum scenes to actually send to VLM per run
VLM_MAX_TOKENS = 48  # shorten generated output to speed up inference
VLM_DEBUG_TIMING = True
SCENE_CONTEXT = {}


def get_vlm_attention_implementation() -> str:
    if DEVICE != "cuda":
        return "sdpa"

    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"

    return "sdpa"


VLM_ATTENTION_IMPLEMENTATION = get_vlm_attention_implementation()


def get_cuda_max_memory() -> dict:
    if DEVICE != "cuda":
        return {}

    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    safe_memory_gb = max(1, int(total_memory_gb) - 1)
    return {0: f"{safe_memory_gb}GiB", "cpu": "0GiB"}

BOUNDARY_COUNT = 10
TOP_K_SELECT = 60
CLUSTER_SIM_THRESHOLD = 0.85
FRAME_CACHE_SIZE = 512
CLIP_INPUT_SIZE = 224

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32)


def can_use_torch_compile() -> bool:
    if not ENABLE_TORCH_COMPILE:
        return False
    if DEVICE != "cuda":
        return False
    if not hasattr(torch, "compile"):
        return False
    return importlib.util.find_spec("triton") is not None


# ----------------------------
# VIDEO CACHE (IMPORTANT FIX)
# ----------------------------
class VideoFrameStore:
    def __init__(self, video_path, cache_size=FRAME_CACHE_SIZE):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.current_pos = 0

    def _store_cache(self, frame_index, frame):
        self.cache[frame_index] = frame
        self.cache.move_to_end(frame_index)
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def get_frame(self, frame_index):
        frame_index = max(0, min(frame_index, max(0, self.frame_count - 1)))

        cached = self.cache.get(frame_index)
        if cached is not None:
            self.cache.move_to_end(frame_index)
            return cached

        if frame_index < self.current_pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            self.current_pos = frame_index
        elif frame_index > self.current_pos:
            forward_steps = frame_index - self.current_pos
            for _ in range(forward_steps):
                if not self.cap.grab():
                    break
            self.current_pos = frame_index

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None

        self._store_cache(frame_index, frame)
        self.current_pos = frame_index + 1
        return frame

    def sample_scene_frames(self, start, end, max_frames=6):
        start_i = int(start * self.fps)
        end_i = int(end * self.fps + 0.5)   # or math.floor
        if end_i <= start_i:
            end_i = start_i + 1

        take = max(1, max_frames)
        if take == 1:
            indices = [start_i]
        else:
            indices = np.linspace(start_i, end_i - 1, num=take, dtype=int).tolist()

        deduped = []
        seen = set()
        for idx in indices:
            if idx not in seen:
                deduped.append(idx)
                seen.add(idx)

        frames = []
        for idx in deduped:
            frame = self.get_frame(idx)
            if frame is not None:
                frames.append(frame)
        return frames

    def close(self):
        self.cap.release()


# ----------------------------
# CLIP ENCODER
# ----------------------------
vision_model = CLIPVisionModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_safetensors=True
).to(DEVICE)

if CLIP_USE_HALF:
    vision_model = vision_model.half()

if ENABLE_TORCH_COMPILE:
    try:
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True
    except Exception:
        pass

if can_use_torch_compile():
    try:
        vision_model = torch.compile(vision_model, mode=TORCH_COMPILE_MODE)
        print(f"CLIP torch.compile enabled (mode={TORCH_COMPILE_MODE})")
    except Exception as compile_e:
        print(f"CLIP torch.compile disabled: {compile_e}")
elif ENABLE_TORCH_COMPILE:
    print("CLIP torch.compile skipped: Triton/compile backend not available on this environment.")

vision_model.eval()


# ----------------------------
# MEMORY
# ----------------------------
class SceneMemory:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)

    def add(self, scene_id, emb):
        self.memory.append((scene_id, emb))

    def max_similarity(self, emb):
        if not self.memory:
            return 0.0
        stacked = torch.stack([x[1] for x in self.memory], dim=0)
        sims = torch.nn.functional.cosine_similarity(
            emb.unsqueeze(0),
            stacked,
            dim=1,
        )
        return float(torch.max(sims).item())



memory = SceneMemory()


# ----------------------------
# LOAD SCENES
# ----------------------------
def load_scenes(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "semantic_scenes" in data:
        return data["semantic_scenes"]
    return data


def scene_id_of(scene, fallback):

    for key in ("index", "scene_id", "id"):
        if key in scene:
            return int(scene[key])

    return int(fallback)


def scene_start_of(scene):
    for key in ("start", "start_seconds", "t_start", "time_start"):
        if key in scene:
            return float(scene[key])
    raise KeyError(f"Scene has no start timestamp: {scene}")


def scene_end_of(scene):
    for key in ("end", "end_seconds", "t_end", "time_end"):
        if key in scene:
            return float(scene[key])
    raise KeyError(f"Scene has no end timestamp: {scene}")


def load_optional_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def load_first_existing_json(paths):
    for path in paths or []:
        data = load_optional_json(path)
        if data is not None:
            return data, path
    return None, None


def overlap_seconds(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def normalize_identity_key(name):
    if name is None:
        return ""

    text = str(name).strip().upper()
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = text.replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def coerce_optional_int(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except Exception:
            return None
    return None


def resolve_reference_frame_path(raw_path, selected_json_path):
    if not raw_path:
        return None

    text = str(raw_path).strip()
    if not text:
        return None

    if os.path.isabs(text):
        return text

    if selected_json_path:
        base_dir = os.path.dirname(selected_json_path)
        if base_dir:
            return os.path.normpath(os.path.join(base_dir, text))

    return text


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def normalize_tokens(text):
    if not text:
        return set()
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return set(words)


def jaccard_distance(tokens_a, tokens_b):
    if not tokens_a and not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    inter = tokens_a & tokens_b
    return 1.0 - (len(inter) / len(union))


def build_script_scene_map(data):
    if not isinstance(data, dict):
        return {}
    scenes = data.get("scenes")
    if not isinstance(scenes, list):
        return {}

    out = {}
    for idx, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue
        scene_id = scene.get("scene_id", idx)
        try:
            scene_id = int(scene_id)
        except Exception:
            scene_id = idx

        heading = scene.get("heading")
        dialogue_blocks = scene.get("dialogue_blocks", [])
        actions = scene.get("action_blocks", [])
        hint_parts = []
        if isinstance(heading, str) and heading.strip():
            hint_parts.append(f"heading: {heading.strip()}")
        if isinstance(dialogue_blocks, list) and dialogue_blocks:
            # keep script hint compact
            preview = []
            for block in dialogue_blocks[:2]:
                if isinstance(block, dict):
                    sp = str(block.get("speaker", "")).strip()
                    tx = str(block.get("text", "")).strip()
                    if sp or tx:
                        preview.append(f"{sp}: {tx}".strip())
            if preview:
                hint_parts.append("script_dialogue: " + " | ".join(preview))
        if isinstance(actions, list) and actions:
            action_preview = [str(a).strip() for a in actions[:1] if str(a).strip()]
            if action_preview:
                hint_parts.append("script_action: " + action_preview[0])

        out[scene_id] = {
            "script_hint": " ; ".join(hint_parts),
        }

    return out


def load_script_scene_map(path):
    return build_script_scene_map(load_optional_json(path))


def build_script_character_catalog(data):
    if not isinstance(data, dict):
        return {}

    groups = OrderedDict()

    def register_character(name, source=None):
        canon = normalize_identity_key(name)
        if not canon:
            return None

        bucket = groups.setdefault(
            canon,
            {
                "character_id": canon,
                "display_name": None,
                "aliases": [],
                "scene_ids": set(),
                "texts": [],
                "sources": [],
            },
        )

        original = str(name).strip()
        if original and original not in bucket["aliases"]:
            bucket["aliases"].append(original)
        if source and source not in bucket["sources"]:
            bucket["sources"].append(source)
        if bucket["display_name"] is None:
            bucket["display_name"] = original
        elif original and "(" not in bucket["display_name"] and "(" in original:
            pass
        elif original and len(original) < len(bucket["display_name"]):
            bucket["display_name"] = original
        return canon

    for raw_name in data.get("characters", []):
        register_character(raw_name, source="character_list")

    scenes = data.get("scenes", [])
    if not isinstance(scenes, list):
        scenes = []

    for idx, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue

        scene_id = scene_id_of(scene, idx)

        for raw_name in scene.get("characters", []) or []:
            canon = register_character(raw_name, source="scene_cast")
            if canon:
                groups[canon]["scene_ids"].add(scene_id)

        dialogue_blocks = scene.get("dialogue_blocks", []) or []
        for block in dialogue_blocks:
            if not isinstance(block, dict):
                continue
            canon = register_character(block.get("speaker"), source="dialogue_block")
            if not canon:
                continue
            text = str(block.get("text", "")).strip()
            if text:
                groups[canon]["texts"].append(text)
            groups[canon]["scene_ids"].add(scene_id)

        action_blocks = scene.get("action_blocks", []) or []
        if isinstance(action_blocks, list):
            scene_text = []
            for item in action_blocks:
                if isinstance(item, str) and item.strip():
                    scene_text.append(item.strip())
            if scene_text:
                scene_blob = " ".join(scene_text)
                for raw_name in scene.get("characters", []) or []:
                    canon = normalize_identity_key(raw_name)
                    if canon in groups:
                        groups[canon]["texts"].append(scene_blob)

    catalog = OrderedDict()
    for canon, bucket in groups.items():
        texts = [t.strip() for t in bucket["texts"] if isinstance(t, str) and t.strip()]
        deduped_texts = []
        seen = set()
        for text in texts:
            if text not in seen:
                deduped_texts.append(text)
                seen.add(text)

        display_name = bucket["display_name"] or canon
        catalog[canon] = {
            "character_id": canon,
            "display_name": display_name,
            "aliases": list(dict.fromkeys(bucket["aliases"])),
            "scene_ids": sorted(bucket["scene_ids"]),
            "profile_text": " ".join(deduped_texts),
            "sources": list(dict.fromkeys(bucket["sources"])),
        }

    return catalog


def get_scene_ranges(scenes):
    ranges = []
    for idx, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue
        try:
            ranges.append((scene_id_of(scene, idx), scene_start_of(scene), scene_end_of(scene)))
        except Exception:
            continue
    return ranges


def span_scene_ids(start, end, scene_ranges):
    ids = []
    for scene_id, scene_start, scene_end in scene_ranges:
        if overlap_seconds(start, end, scene_start, scene_end) > 0:
            ids.append(scene_id)
    return ids


def build_text_embeddings(texts):
    import numpy as np

    prepared = [text if isinstance(text, str) and text.strip() else " " for text in texts]

    if len(prepared) == 0:
        return np.zeros((0, 0), dtype=np.float32), "empty"

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(TEXT_EMBEDDING_MODEL, device="cpu")

    embeddings = model.encode(
        prepared,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    print(f"[embed] sentence-transformers output shape: {embeddings.shape}")
    assert embeddings.shape[0] == len(texts)

    return embeddings, "sentence-transformers"
    
def cosine_similarity_matrix(a, b):
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return np.clip(a @ b.T, -1.0, 1.0)


def build_identity_reconciliation(scenes, whisper_words, script_data):
    if not ENABLE_IDENTITY_RECONCILIATION:
        return {
            "enabled": False,
            "script_characters": {},
            "speaker_profiles": {},
            "face_cluster_profiles": {},
            "speaker_to_character": {},
            "face_cluster_to_character": {},
            "unresolved_speakers": [],
            "unresolved_face_clusters": [],
            "unified_characters": [],
        }

    scene_ranges = get_scene_ranges(scenes)
    script_characters = build_script_character_catalog(script_data)
    selected_reference_clusters, selected_reference_clusters_path = load_first_existing_json(SELECTED_REFERENCE_CLUSTERS_PATHS)
    selected_reference_cluster_ids = set()
    selected_reference_entries = []
    if isinstance(selected_reference_clusters, list):
        for item in selected_reference_clusters:
            if isinstance(item, dict):
                cluster_id = coerce_optional_int(item.get("cluster"))
                character_id = normalize_identity_key(item.get("character_id"))
                character_ids = [
                    normalize_identity_key(x)
                    for x in (item.get("character_ids") or [])
                    if normalize_identity_key(x)
                ]
                if not character_id and character_ids:
                    character_id = character_ids[0]
                representative_frame = resolve_reference_frame_path(
                    item.get("representative_frame"),
                    selected_reference_clusters_path,
                )
            else:
                cluster_id = coerce_optional_int(item)
                character_id = ""
                character_ids = []
                representative_frame = None
            if cluster_id is not None:
                selected_reference_cluster_ids.add(cluster_id)
                target_ids = character_ids if character_ids else [character_id]
                for cid in target_ids:
                    if not cid:
                        continue
                    selected_reference_entries.append(
                        {
                            "cluster": cluster_id,
                            "character_id": cid,
                            "representative_frame": representative_frame,
                        }
                    )

    selected_reference_entries = [entry for entry in selected_reference_entries if entry.get("character_id")]
    selected_reference_embeddings = OrderedDict()
    for entry in selected_reference_entries:
        frame_path = entry.get("representative_frame")
        emb = compute_face_embedding_from_image_path(frame_path)
        if emb is None:
            continue
        selected_reference_embeddings.setdefault(entry["character_id"], []).append(emb)

    selected_reference_embeddings = {
        character_id: np.mean(np.stack(embs, axis=0), axis=0)
        for character_id, embs in selected_reference_embeddings.items()
        if embs
    }
    for character_id, emb in list(selected_reference_embeddings.items()):
        norm = np.linalg.norm(emb)
        if norm > 0:
            selected_reference_embeddings[character_id] = emb / norm

    whisperx_face_segments = load_whisperx_face_segments(WHISPERX_JSON_PATH)
    whisperx_face_embeddings = OrderedDict((seg["cluster"], seg["embedding"]) for seg in whisperx_face_segments)
    if selected_reference_cluster_ids and not selected_reference_embeddings:
        print(
            f"[identity] selected reference clusters were loaded, but no labeled face-image embeddings could be built. "
            f"Fill character_id in the report selection and ensure representative_frame paths are valid."
        )

    speaker_profiles = OrderedDict()
    face_cluster_profiles = OrderedDict()

    for word in whisper_words:
        speaker = str(word.get("speaker", "")).strip()
        if not speaker:
            continue
        cluster = coerce_optional_int(word.get("character_cluster"))

        start = safe_float(word.get("start"))
        end = safe_float(word.get("end"))
        if start is None or end is None:
            continue

        scene_ids = span_scene_ids(start, end, scene_ranges)
        token = str(word.get("word", "")).strip()

        speaker_bucket = speaker_profiles.setdefault(
            speaker,
            {
                "speaker_id": speaker,
                "scene_ids": set(),
                "texts": [],
                "word_count": 0,
                "face_clusters": Counter(),
            },
        )
        speaker_bucket["scene_ids"].update(scene_ids)
        if token:
            speaker_bucket["texts"].append(token)
        speaker_bucket["word_count"] += 1
        if cluster is not None:
            speaker_bucket["face_clusters"][cluster] += 1

        if cluster is None:
            continue

        face_bucket = face_cluster_profiles.setdefault(
            cluster,
            {
                "face_cluster_id": cluster,
                "scene_ids": set(),
                "texts": [],
                "word_count": 0,
                "speakers": Counter(),
            },
        )
        face_bucket["scene_ids"].update(scene_ids)
        if token:
            face_bucket["texts"].append(token)
        face_bucket["word_count"] += 1
        face_bucket["speakers"][speaker] += 1

    speaker_ids = list(speaker_profiles.keys())
    character_ids = list(script_characters.keys())
    face_ids = list(face_cluster_profiles.keys())

    # selected_reference_cluster_ids are only used to build labeled visual reference embeddings.
    # They should not filter the observed face clusters coming from whisperx_results.json.

    speaker_texts = [" ".join(profile["texts"]) for profile in speaker_profiles.values()]
    character_texts = [script_characters[cid]["profile_text"] for cid in character_ids]
    face_texts = [" ".join(profile["texts"]) for profile in face_cluster_profiles.values()]

    speaker_embeds, embed_method = build_text_embeddings(speaker_texts)
    character_embeds, _ = build_text_embeddings(character_texts)
    face_embeds, _ = build_text_embeddings(face_texts)

    total_scenes = max(1, len(scene_ranges))
    total_words = max(1, len(whisper_words))

    speaker_stability_scores = {}
    for speaker_id, profile in speaker_profiles.items():
        scene_count = len(profile["scene_ids"])
        word_count = int(profile["word_count"])
        stability = 0.60 * min(1.0, scene_count / total_scenes) + 0.40 * min(1.0, word_count / total_words)
        speaker_stability_scores[speaker_id] = round(float(stability), 6)

    speaker_to_character = {}
    face_cluster_to_character = {}
    speaker_character_scores = {}
    face_character_scores = {}
    unresolved_speakers = []
    unresolved_face_clusters = []

    if speaker_ids and character_ids and speaker_embeds.size and character_embeds.size:
        text_sim = cosine_similarity_matrix(speaker_embeds, character_embeds)
        scene_sim = np.zeros_like(text_sim)
        for i, speaker_id in enumerate(speaker_ids):
            speaker_scenes = set(speaker_profiles[speaker_id]["scene_ids"])
            for j, character_id in enumerate(character_ids):
                character_scenes = set(script_characters[character_id]["scene_ids"])
                scene_sim[i, j] = 1.0 - jaccard_distance(speaker_scenes, character_scenes)

        speaker_anchor = scene_sim.copy()
        speaker_weak = text_sim
        combined = 0.70 * speaker_anchor + 0.30 * speaker_weak

        strong_speaker_rows = []
        for i, speaker_id in enumerate(speaker_ids):
            profile = speaker_profiles[speaker_id]
            word_count = int(profile["word_count"])
            # 只要出现至少 1 个词就尝试匹配（可调整）
            if word_count >= 1:
                strong_speaker_rows.append(i)

        if strong_speaker_rows:
            cost = -speaker_anchor[strong_speaker_rows, :]
            row_ind, col_ind = linear_sum_assignment(cost)
            for local_r, c in zip(row_ind, col_ind):
                r = strong_speaker_rows[local_r]
                speaker_id = speaker_ids[r]
                anchor_score = float(speaker_anchor[r, c])
                weak_score = float(speaker_weak[r, c])
                stability = speaker_stability_scores.get(speaker_id, 0.0)
                confidence = 0.75 * anchor_score + 0.25 * weak_score
                confidence *= 0.60 + 0.40 * stability
                if anchor_score < 0.15 or confidence < SPEAKER_CHARACTER_MIN_SIM:
                    unresolved_speakers.append(
                        {
                            "speaker_id": speaker_id,
                            "best_character_id": character_ids[c],
                            "confidence": round(confidence, 6),
                            "anchor_score": round(anchor_score, 6),
                            "weak_score": round(weak_score, 6),
                            "stability": round(stability, 6),
                        }
                    )
                    continue
                character_id = character_ids[c]
                speaker_to_character[speaker_id] = character_id
                speaker_character_scores[speaker_id] = round(confidence, 6)

        for i, speaker_id in enumerate(speaker_ids):
            if speaker_id in speaker_to_character:
                continue
            profile = speaker_profiles[speaker_id]
            scene_count = len(profile["scene_ids"])
            word_count = int(profile["word_count"])
            stability = speaker_stability_scores.get(speaker_id, 0.0)
            scores = combined[i]
            if scores.size == 0:
                unresolved_speakers.append(
                    {
                        "speaker_id": speaker_id,
                        "best_character_id": None,
                        "confidence": 0.0,
                        "anchor_score": 0.0,
                        "weak_score": 0.0,
                        "stability": round(stability, 6),
                    }
                )
                continue
            best_index = int(np.argmax(scores))
            anchor_score = float(speaker_anchor[i, best_index])
            weak_score = float(speaker_weak[i, best_index])
            confidence = float(scores[best_index]) * (0.60 + 0.40 * stability)
            if scene_count >= 2 and word_count >= 4 and anchor_score >= 0.20 and confidence >= SPEAKER_CHARACTER_MIN_SIM:
                character_id = character_ids[best_index]
                speaker_to_character[speaker_id] = character_id
                speaker_character_scores[speaker_id] = round(confidence, 6)
            else:
                unresolved_speakers.append(
                    {
                        "speaker_id": speaker_id,
                        "best_character_id": character_ids[best_index],
                        "confidence": round(confidence, 6),
                        "anchor_score": round(anchor_score, 6),
                        "weak_score": round(weak_score, 6),
                        "stability": round(stability, 6),
                    }
                )
                
    # Added: Include all speakers in the unresolved list in the mapping (provided best_character_id is valid)
    for u in unresolved_speakers:
        spk = u.get("speaker_id")
        best_char = u.get("best_character_id")
        if spk and best_char and spk != "None" and best_char != "None":
            if spk not in speaker_to_character:
                speaker_to_character[spk] = best_char
                speaker_character_scores[spk] = u.get("confidence", 0.0)  # 保留原置信度供参考

    if face_ids:
        face_visual_text_support = np.zeros((len(face_ids), len(character_ids)), dtype=np.float32)
        scene_sim = np.zeros((len(face_ids), len(character_ids)), dtype=np.float32)
        speaker_support = np.zeros((len(face_ids), len(character_ids)), dtype=np.float32)

        character_index_map = {character_id: idx for idx, character_id in enumerate(character_ids)}
        for i, face_id in enumerate(face_ids):
            face_scenes = set(face_cluster_profiles[face_id]["scene_ids"])
            for j, character_id in enumerate(character_ids):
                character_scenes = set(script_characters[character_id]["scene_ids"])
                scene_sim[i, j] = 1.0 - jaccard_distance(face_scenes, character_scenes)

            speaker_counts = face_cluster_profiles[face_id]["speakers"]
            total_weight = sum(speaker_counts.values()) or 1.0
            for speaker_id, weight in speaker_counts.items():
                mapped_character = speaker_to_character.get(speaker_id)
                if mapped_character is None:
                    continue
                char_index = character_index_map[mapped_character]
                speaker_support[i, char_index] += float(weight) / float(total_weight)

            # visual reference support: compare target face cluster embedding to selected reference embeddings
            target_emb = whisperx_face_embeddings.get(face_id)
            if target_emb is not None and selected_reference_embeddings:
                for j, character_id in enumerate(character_ids):
                    ref_emb = selected_reference_embeddings.get(character_id)
                    if ref_emb is None:
                        continue
                    face_visual_text_support[i, j] = float(np.clip(np.dot(target_emb, ref_emb), -1.0, 1.0))

        if selected_reference_embeddings:
            combined = 0.60 * face_visual_text_support + 0.15 * scene_sim + 0.25 * speaker_support
        else:
            # fallback to the original text-driven face mapping if no visual references are available
            text_sim = cosine_similarity_matrix(face_embeds, character_embeds)
            combined = 0.30 * text_sim + 0.25 * scene_sim + 0.45 * speaker_support

        for i, face_id in enumerate(face_ids):
            scores = combined[i]
            if scores.size == 0:
                unresolved_face_clusters.append(
                    {
                        "face_cluster_id": face_id,
                        "best_character_id": None,
                        "confidence": 0.0,
                        "speaker_support": 0.0,
                        "scene_support": 0.0,
                        "visual_support": 0.0,
                    }
                )
                continue
            best_index = int(np.argmax(scores))
            score = float(scores[best_index])
            scene_score = float(scene_sim[i, best_index])
            speaker_score = float(speaker_support[i, best_index])
            visual_score = float(face_visual_text_support[i, best_index])
            confident_support = visual_score >= 0.20 or speaker_score >= 0.20 or scene_score >= 0.25
            if score < FACE_CHARACTER_MIN_SIM or not confident_support:
                unresolved_face_clusters.append(
                    {
                        "face_cluster_id": face_id,
                        "best_character_id": character_ids[best_index],
                        "confidence": round(score, 6),
                        "speaker_support": round(speaker_score, 6),
                        "scene_support": round(scene_score, 6),
                        "visual_support": round(visual_score, 6),
                    }
                )
                continue
            face_cluster_to_character[face_id] = character_ids[best_index]
            face_character_scores[face_id] = round(score, 6)


    unified_characters = []
    node_lookup = {}
    for character_id, profile in script_characters.items():
        node = {
            "character_id": character_id,
            "display_name": profile["display_name"],
            "aliases": profile["aliases"],
            "scene_ids": profile["scene_ids"],
            "profile_text": profile["profile_text"],
            "speakers": [],
            "face_clusters": [],
        }
        node_lookup[character_id] = node
        unified_characters.append(node)

    for speaker_id, character_id in speaker_to_character.items():
        node = node_lookup.get(character_id)
        if node is None:
            continue
        node["speakers"].append(
            {
                "speaker_id": speaker_id,
                "score": speaker_character_scores.get(speaker_id, 0.0),
            }
        )

    for face_id, character_id in face_cluster_to_character.items():
        node = node_lookup.get(character_id)
        if node is None:
            continue
        node["face_clusters"].append(
            {
                "face_cluster_id": face_id,
                "score": face_character_scores.get(face_id, 0.0),
            }
        )

    for node in unified_characters:
        node["speakers"] = sorted(node["speakers"], key=lambda item: item["score"], reverse=True)
        node["face_clusters"] = sorted(node["face_clusters"], key=lambda item: item["score"], reverse=True)

    return {
        "enabled": True,
        "embedding_method": embed_method,
        "selected_reference_clusters_path": selected_reference_clusters_path,
        "selected_reference_cluster_ids": sorted(selected_reference_cluster_ids),
        "selected_reference_embedding_count": len(selected_reference_embeddings),
        "script_characters": script_characters,
        "speaker_profiles": speaker_profiles,
        "face_cluster_profiles": face_cluster_profiles,
        "speaker_to_character": speaker_to_character,
        "speaker_character_scores": speaker_character_scores,
        "face_cluster_to_character": face_cluster_to_character,
        "face_character_scores": face_character_scores,
        "speaker_stability_scores": speaker_stability_scores,
        "unresolved_speakers": unresolved_speakers,
        "unresolved_face_clusters": unresolved_face_clusters,
        "unified_characters": unified_characters,
    }


def attach_identity_links_to_record(record, identity_map):
    speaker_to_character = identity_map.get("speaker_to_character", {})
    face_to_character = identity_map.get("face_cluster_to_character", {})

    mapped_speakers = []
    for speaker in record.get("speakers", []):
        character_id = speaker_to_character.get(str(speaker))
        if character_id and character_id not in mapped_speakers:
            mapped_speakers.append(character_id)

    mapped_faces = []
    for face_cluster in record.get("face_clusters", []):
        face_id = coerce_optional_int(face_cluster)
        if face_id is None:
            continue
        character_id = face_to_character.get(face_id)
        if character_id and character_id not in mapped_faces:
            mapped_faces.append(character_id)

    record["speaker_character_ids"] = mapped_speakers
    record["face_character_ids"] = mapped_faces
    record["unified_character_ids"] = sorted(set(mapped_speakers) | set(mapped_faces))


def identity_summary_payload(identity_map):
    if not identity_map.get("enabled"):
        return {"enabled": False}

    return {
        "enabled": True,
        "embedding_method": identity_map.get("embedding_method"),
        "selected_reference_clusters_path": identity_map.get("selected_reference_clusters_path"),
        "selected_reference_cluster_ids": identity_map.get("selected_reference_cluster_ids", []),
        "selected_reference_embedding_count": identity_map.get("selected_reference_embedding_count", 0),
        "speaker_to_character": identity_map.get("speaker_to_character", {}),
        "speaker_character_scores": identity_map.get("speaker_character_scores", {}),
        "face_cluster_to_character": identity_map.get("face_cluster_to_character", {}),
        "face_character_scores": identity_map.get("face_character_scores", {}),
        "speaker_stability_scores": identity_map.get("speaker_stability_scores", {}),
        "unresolved_speakers": identity_map.get("unresolved_speakers", []),
        "unresolved_face_clusters": identity_map.get("unresolved_face_clusters", []),
        "unified_characters": identity_map.get("unified_characters", []),
    }


def collect_whisperx_words(path):
    data = load_optional_json(path)
    if not isinstance(data, dict):
        return []

    # Prioritize word-level results (if they exist)
    words = data.get("words")
    if isinstance(words, list):
        normalized = []
        for w in words:
            if not isinstance(w, dict):
                continue
            start = safe_float(w.get("start"))
            end = safe_float(w.get("end"))
            if start is None or end is None:
                continue
            if end <= start:
                continue
            token = str(w.get("word", "")).strip()
            if not token:
                continue
            speaker = w.get("speaker")
            face_cluster = w.get("character_cluster")
            normalized.append(
                {
                    "start": start,
                    "end": end,
                    "word": token,
                    "speaker": str(speaker) if speaker is not None else None,
                    "character_cluster": coerce_optional_int(face_cluster),
                }
            )
        return normalized

    # Otherwise, try to construct (sentence-level) from the "result" field.
    result = data.get("result")
    if not isinstance(result, list):
        return []

    normalized = []
    for seg in result:
        if not isinstance(seg, dict):
            continue
        start = safe_float(seg.get("start"))
        end = safe_float(seg.get("end"))
        if start is None or end is None:
            continue
        if end <= start:
            continue
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        speaker = seg.get("speaker")
       
        character_cluster = seg.get("character_cluster") or seg.get("character")
        normalized.append(
            {
                "start": start,
                "end": end,
                "word": text,         # Store the entire sentence as a "word" field
                "speaker": str(speaker) if speaker is not None else None,
                "character_cluster": coerce_optional_int(character_cluster),
            }
        )
    return normalized


def load_whisperx_face_segments(path):
    data = load_optional_json(path)
    if not isinstance(data, dict):
        return []

    face_segments = data.get("faces")
    if not isinstance(face_segments, list):
        return []

    normalized = []
    for item in face_segments:
        if not isinstance(item, dict):
            continue
        cluster_id = coerce_optional_int(item.get("cluster"))
        embedding = item.get("embedding")
        if cluster_id is None or not isinstance(embedding, list) or not embedding:
            continue
        try:
            emb = np.asarray(embedding, dtype=np.float32)
        except Exception:
            continue
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        normalized.append(
            {
                "cluster": cluster_id,
                "start": safe_float(item.get("start")),
                "end": safe_float(item.get("end")),
                "embedding": emb,
                "num_samples": int(item.get("num_samples", 0) or 0),
                "frame_indices": item.get("frame_indices", []),
                "gender": item.get("gender"),
            }
        )

    return normalized


_INSIGHTFACE_MODEL = None


def get_insightface_model():
    global _INSIGHTFACE_MODEL
    if _INSIGHTFACE_MODEL is not None:
        return _INSIGHTFACE_MODEL

    try:
        import insightface
    except Exception:
        return None

    try:
        model = insightface.app.FaceAnalysis()
        model.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
        _INSIGHTFACE_MODEL = model
        return _INSIGHTFACE_MODEL
    except Exception as e:
        print(f"[identity] insightface unavailable for visual refs: {e}")
        return None


def compute_face_embedding_from_image_path(image_path):
    model = get_insightface_model()
    if model is None:
        return None

    if not image_path:
        return None

    if not os.path.exists(image_path):
        return None

    image = cv2.imread(image_path)
    if image is None:
        return None

    try:
        faces = model.get(image)
    except Exception:
        return None

    if not faces:
        return None

    def face_area(face):
        bbox = getattr(face, "bbox", None)
        if bbox is None:
            return 0.0
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    face = max(faces, key=face_area)
    embedding = getattr(face, "embedding", None)
    if embedding is None:
        return None

    emb = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def enrich_record_with_text_face(record, whisper_words, script_scene_map):
    start = record["start"]
    end = record["end"]

    scene_words = []
    speakers = []
    face_clusters = []

    for w in whisper_words:
        if overlap_seconds(start, end, w["start"], w["end"]) <= 0:
            continue
        scene_words.append(w["word"])
        sp = w.get("speaker")
        if sp and sp not in speakers:
            speakers.append(sp)
        fc = w.get("character_cluster")
        if fc is not None and fc not in face_clusters:
            face_clusters.append(fc)

    scene_text = " ".join(scene_words).strip()
    record["transcript_text"] = scene_text
    record["speakers"] = speakers
    record["face_clusters"] = face_clusters
    record["transcript_tokens"] = normalize_tokens(scene_text)

    script_hint = ""
    script_entry = script_scene_map.get(record["scene_id"])
    if isinstance(script_entry, dict):
        script_hint = str(script_entry.get("script_hint", "")).strip()
    record["script_hint"] = script_hint


def apply_multimodal_boundary_priors(records):
    if not records:
        return

    records[0]["text_boundary_prior"] = 0.0
    records[0]["face_boundary_prior"] = 0.0

    for i in range(1, len(records)):
        prev = records[i - 1]
        cur = records[i]

        text_shift = jaccard_distance(
            prev.get("transcript_tokens", set()),
            cur.get("transcript_tokens", set()),
        )

        prev_faces = set(prev.get("face_clusters", []))
        cur_faces = set(cur.get("face_clusters", []))
        face_shift = jaccard_distance(prev_faces, cur_faces)

        cur["text_boundary_prior"] = round(float(text_shift), 6)
        cur["face_boundary_prior"] = round(float(face_shift), 6)

    for rec in records:
        fused = (
            rec["importance_score"]
            + TEXT_BOUNDARY_WEIGHT * rec.get("text_boundary_prior", 0.0)
            + FACE_BOUNDARY_WEIGHT * rec.get("face_boundary_prior", 0.0)
        )
        rec["unified_score"] = round(float(fused), 6)


# ----------------------------

# ----------------------------
# CLIP EMBEDDING
# ----------------------------
def extract_embedding(frames):
    if not frames:
        return None

    pixel_values = preprocess_clip_batch_torch(frames)

    with torch.no_grad():
        feats = vision_model(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        emb = feats.mean(dim=0)
        emb = torch.nn.functional.normalize(emb, dim=0)

    return emb.float().cpu()


def preprocess_clip_batch_torch(frames):
    frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    frame_batch = np.stack(frames, axis=0)
    tensor = torch.from_numpy(frame_batch)
    tensor = tensor.to(DEVICE, non_blocking=True)
    tensor = tensor.permute(0, 3, 1, 2).float() / 255.0
    tensor = tensor[:, [2, 1, 0], :, :]
    tensor = torch.nn.functional.interpolate(
        tensor,
        size=(CLIP_INPUT_SIZE, CLIP_INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    mean = CLIP_MEAN.to(tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)
    std = CLIP_STD.to(tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    if CLIP_USE_HALF:
        tensor = tensor.half()

    return tensor.contiguous()


def extract_embeddings_batched(scene_frames):
    stage_start = time.perf_counter()
    flat = []
    slices = []
    offset = 0
    for frames in scene_frames:
        slices.append((offset, len(frames)))
        flat.extend(frames)
        offset += len(frames)

    if not flat:
        return [None for _ in scene_frames]

    all_embeds = []
    for batch_index, i in enumerate(range(0, len(flat), CLIP_BATCH_SIZE), start=1):
        batch_start = time.perf_counter()
        batch_frames = flat[i:i + CLIP_BATCH_SIZE]
        pixel_values = preprocess_clip_batch_torch(batch_frames)

        with torch.no_grad():
            feats = vision_model(pixel_values=pixel_values).last_hidden_state[:, 0, :]
            feats = torch.nn.functional.normalize(feats, dim=1)
        all_embeds.append(feats.float().cpu())
        log_timing(f"CLIP batch {batch_index} ({len(batch_frames)} frames)", batch_start)

    all_embeds = torch.cat(all_embeds, dim=0)

    scene_embeds = []
    for start, length in slices:
        if length <= 0:
            scene_embeds.append(None)
            continue
        emb = all_embeds[start:start + length].mean(dim=0)
        emb = torch.nn.functional.normalize(emb, dim=0)
        scene_embeds.append(emb)
    log_timing(f"CLIP embedding for {len(scene_frames)} scenes", stage_start)
    return scene_embeds


def load_vlm_cache(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return {}


def save_vlm_cache(path, cache_data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)


def log_timing(label, start_time):
    elapsed = time.perf_counter() - start_time
    print(f"[timing] {label}: {elapsed:.3f}s")


def parse_structured_output(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {
            "summary": "",
            "actions": [],
            "raw_text": text.strip(),
            "relation_state": "",
        }

    candidate = match.group(0)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {
            "summary": "",
            "actions": [],
            "raw_text": text.strip(),
            "relation_state": "",
        }

    return {
        "summary": parsed.get("summary", ""),
        "actions": parsed.get("actions", []),
        "relation_state": parsed.get("relation_state", ""),
    }


def collect_scene_character_names(record):
    names = []
    for key in ("unified_character_ids", "face_character_ids", "speaker_character_ids"):
        for value in record.get(key, []) or []:
            name = str(value).strip()
            if name and name not in names:
                names.append(name)
    return names


def rewrite_person_mentions(text, character_names):
    if not isinstance(text, str) or not text.strip() or not character_names:
        return text

    primary = character_names[0]
    secondary = character_names[1] if len(character_names) > 1 else primary

    replacements = [
        (r"\bA\s+man\b", primary),
        (r"\bA\s+woman\b", secondary),
        (r"\bThe\s+man\b", primary),
        (r"\bThe\s+woman\b", secondary),
        (r"\ba\s+man\b", primary),
        (r"\ba\s+woman\b", secondary),
        (r"\bman\b", primary),
        (r"\bwoman\b", secondary),
        (r"\bboy\b", primary),
        (r"\bgirl\b", secondary),
        (r"\bguy\b", primary),
        (r"\blady\b", secondary),
        (r"\bperson\b", primary),
    ]

    rewritten = text
    for pattern, replacement in replacements:
        rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
    return rewritten


def normalize_vlm_description(description, record):
    if not isinstance(description, dict):
        return description

    character_names = collect_scene_character_names(record)
    if not character_names:
        return description

    normalized = dict(description)
    for key in ("summary", "relation_state", "raw_text"):
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = rewrite_person_mentions(value, character_names)

    actions = normalized.get("actions")
    if isinstance(actions, list):
        normalized["actions"] = [
            rewrite_person_mentions(action, character_names) if isinstance(action, str) else action
            for action in actions
        ]

    return normalized


# ----------------------------
# LOAD LLM
# ----------------------------
llm_model = None
llm_processor = None

if ENABLE_LLM:
    try:
        if Qwen2VLForConditionalGeneration is None or AutoProcessor is None:
            raise RuntimeError("Qwen2-VL classes are unavailable in transformers build")

        print(f"Loading VLM: {MULTIMODAL_MODEL}")

        llm_processor = AutoProcessor.from_pretrained(MULTIMODAL_MODEL)

        model_kwargs = {
            "dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
            "attn_implementation": VLM_ATTENTION_IMPLEMENTATION,
            "low_cpu_mem_usage": True,
        }

        if DEVICE == "cuda" and PREFER_FAST_SINGLE_GPU:
            try:
                llm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    MULTIMODAL_MODEL,
                    **model_kwargs,
                )
                llm_model = llm_model.to("cuda")
            except RuntimeError as inner_e:
                is_oom = "out of memory" in str(inner_e).lower()
                if not (FALLBACK_TO_AUTO_ON_OOM and is_oom):
                    raise
                torch.cuda.empty_cache()
                if BitsAndBytesConfig is None or not ENABLE_4BIT_FALLBACK:
                    raise RuntimeError(
                        "Qwen2-VL-7B does not fit in GPU memory in fp16 without CPU offload. "
                        "Install/enable 4bit quantization or use a smaller model."
                    ) from inner_e

                print("Fast single-GPU load OOM, falling back to 4bit GPU-only loading.")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                llm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    MULTIMODAL_MODEL,
                    quantization_config=quantization_config,
                    attn_implementation=VLM_ATTENTION_IMPLEMENTATION,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    max_memory=get_cuda_max_memory(),
                )
        else:
            llm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                MULTIMODAL_MODEL,
                **model_kwargs,
            )
            if DEVICE != "cuda":
                llm_model = llm_model.to(DEVICE)
        if hasattr(llm_model, "hf_device_map"):
            print(f"VLM hf_device_map: {llm_model.hf_device_map}")
            if any(device == "cpu" for device in llm_model.hf_device_map.values()):
                raise RuntimeError(
                    "Model loaded with CPU offload. Disable auto placement or use a smaller / quantized model."
                )
        else:
            print("VLM hf_device_map: <none>")
        try:
            print(f"VLM first parameter device: {next(llm_model.parameters()).device}")
        except Exception:
            pass
        llm_model.eval()

    except Exception as e:
        print("LLM load failed:", e)
        ENABLE_LLM = False


# ----------------------------
# VLM INFERENCE (FIXED MULTI-FRAME INPUT)
# ----------------------------
def describe_scene(frames, scene_id, start, end):
    if not ENABLE_LLM or len(frames) == 0:
        return {
            "scene_id": scene_id,
            "description": {
                "summary": "",
                "actions": [],
                "relation_state": "",
            },
            "start": start,
            "end": end,
        }

    images = [
        Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        for f in frames
    ]

    prompt = (
        "Return ONLY compact JSON with keys summary, actions, relation_state. "
        "Keep summary under 25 words, actions to at most 2 items, and avoid extra keys or markdown."
    )
    
    #context_lines = []
    #ctx = SCENE_CONTEXT
    #if rec_transcript_text := ctx.get("transcript_text", ""):
    #    context_lines.append(f"transcript: {rec_transcript_text[:400]}")
    #if rec_speakers := ctx.get("speakers", []):
    #    context_lines.append("speakers: " + ", ".join(map(str, rec_speakers[:6])))
    #if rec_faces := ctx.get("face_clusters", []):
    #    context_lines.append("face_clusters: " + ", ".join(map(str, rec_faces[:6])))
    #if rec_identity_ids := ctx.get("unified_character_ids", []):
    #    context_lines.append("unified_character_ids: " + ", ".join(map(str, rec_identity_ids[:6])))
    #    context_lines.append("prefer these character labels instead of generic person labels when describing visible people")
    #if rec_script_hint := ctx.get("script_hint", ""):
    #    context_lines.append(f"script_hint: {rec_script_hint[:300]}")

    #if context_lines:
    #    prompt = prompt + "\nUse the following context as anchor evidence (do not invent missing facts):\n" + "\n".join(context_lines)

    preprocess_start = time.perf_counter()
    
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image"} for _ in images],
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    text = llm_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    processor_start = time.perf_counter()

    inputs = llm_processor(
        text=text,
        images=images,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(next(llm_model.parameters()).device)
        if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    transfer_done = time.perf_counter()

    with torch.no_grad():
        out = llm_model.generate(
            **inputs,
            max_new_tokens=VLM_MAX_TOKENS,
            do_sample=False,
        )

    generate_done = time.perf_counter()

    generated = out
    if "input_ids" in inputs:
        generated = out[:, inputs["input_ids"].shape[1]:]
    decoded = llm_processor.batch_decode(generated, skip_special_tokens=True)[0]
    decode_done = time.perf_counter()

    if VLM_DEBUG_TIMING:
        print(
            f"[VLM timing] scene {scene_id} prep={processor_start - preprocess_start:.3f}s "
            f"processor={transfer_done - processor_start:.3f}s generate={generate_done - transfer_done:.3f}s "
            f"decode={decode_done - generate_done:.3f}s total={decode_done - preprocess_start:.3f}s"
        )

    structured = parse_structured_output(decoded)

    return {
        "scene_id": scene_id,
        "start": start,
        "end": end,
        "description": structured,
    }


def compute_importance(records):
    local_memory = SceneMemory()
    for rec in records:
        emb = rec["embedding"]
        max_sim = local_memory.max_similarity(emb)
        score = 1.0 - max_sim
        rec["max_similarity_to_memory"] = round(max_sim, 6)
        rec["importance_score"] = round(float(score), 6)
        rec["unified_score"] = rec["importance_score"]
        local_memory.add(rec["scene_id"], emb)


def select_scene_positions(records):
    count = len(records)
    if count == 0:
        return [], {}

    tags = {i: set() for i in range(count)}

    for i in range(min(BOUNDARY_COUNT, count)):
        tags[i].add("boundary")
    for i in range(max(0, count - BOUNDARY_COUNT), count):
        tags[i].add("boundary")

    ranked = sorted(
        range(count),
        key=lambda i: records[i]["unified_score"],
        reverse=True,
    )
    for rank, i in enumerate(ranked, start=1):
        records[i]["score_rank"] = rank

    for i in ranked[: min(TOP_K_SELECT, count)]:
        tags[i].add("top_k")

    selected_positions = sorted([i for i, t in tags.items() if t])
    return selected_positions, tags


def contiguous_scene_segments(records):
    segments = []
    if not records:
        return segments

    current = {
        "start_pos": 0,
        "end_pos": 0,
        "centroid": records[0]["embedding"].clone(),
        "scores": [records[0]["importance_score"]],
    }

    for pos in range(1, len(records)):
        emb = records[pos]["embedding"]
        sim = torch.nn.functional.cosine_similarity(
            current["centroid"].unsqueeze(0),
            emb.unsqueeze(0),
            dim=1,
        ).item()

        if sim >= CLUSTER_SIM_THRESHOLD:
            n = current["end_pos"] - current["start_pos"] + 1
            current["centroid"] = (current["centroid"] * n + emb) / (n + 1)
            current["end_pos"] = pos
            current["scores"].append(records[pos]["importance_score"])
        else:
            segments.append(current)
            current = {
                "start_pos": pos,
                "end_pos": pos,
                "centroid": emb.clone(),
                "scores": [records[pos]["importance_score"]],
            }

    segments.append(current)

    output = []
    for seg_id, seg in enumerate(segments, start=1):
        s_pos = seg["start_pos"]
        e_pos = seg["end_pos"]
        output.append(
            {
                "segment_id": seg_id,
                "scene_count": e_pos - s_pos + 1,
                "start_scene_id": records[s_pos]["scene_id"],
                "end_scene_id": records[e_pos]["scene_id"],
                "start_time": records[s_pos]["start"],
                "end_time": records[e_pos]["end"],
                "avg_importance": round(float(np.mean(seg["scores"])), 6),
            }
        )

    return output


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():
    total_start = time.perf_counter()
    scenes = load_scenes("semantic_scenes.json")
    frame_store = VideoFrameStore(video_path=VIDEO_PATH)

    records = []
    whisper_words = collect_whisperx_words(WHISPERX_JSON_PATH) if ENABLE_MULTIMODAL_PRIOR else []
    script_data = load_optional_json(SCRIPT_JSON_PATH) if ENABLE_MULTIMODAL_PRIOR else None
    script_scene_map = build_script_scene_map(script_data) if ENABLE_MULTIMODAL_PRIOR else {}
    identity_map = build_identity_reconciliation(scenes, whisper_words, script_data) if ENABLE_MULTIMODAL_PRIOR else {"enabled": False}

    if ENABLE_MULTIMODAL_PRIOR:
        print(f"[prior] whisper words: {len(whisper_words)} from {WHISPERX_JSON_PATH}")
        print(f"[prior] script scene hints: {len(script_scene_map)} from {SCRIPT_JSON_PATH}")
    if identity_map.get("enabled"):
        speaker_map = identity_map.get("speaker_to_character")
        face_map = identity_map.get("face_cluster_to_character")
        speaker_count = len(speaker_map) if isinstance(speaker_map, dict) else 0
        face_count = len(face_map) if isinstance(face_map, dict) else 0
        print(
            f"[identity] speaker mappings: {speaker_count}, "
            f"face mappings: {face_count}"
        )
        selected_ref_ids = identity_map.get("selected_reference_cluster_ids", [])
        selected_ref_path = identity_map.get("selected_reference_clusters_path")
        if selected_ref_ids:
            print(
                f"[identity] selected reference clusters loaded: {len(selected_ref_ids)} "
                f"from {selected_ref_path}"
            )

    scene_clip_frames = []
    sampling_start = time.perf_counter()
    for idx, s in enumerate(scenes, start=1):
        scene_id = scene_id_of(s, idx)
        start = scene_start_of(s)
        end = scene_end_of(s)

        frames = frame_store.sample_scene_frames(start, end, MAX_FRAMES_PER_SCENE)
        if len(frames) == 0:
            continue

        scene_clip_frames.append(frames)

        records.append(
            {
                "scene_id": scene_id,
                "start": float(start),
                "end": float(end),
            }
        )
    log_timing(f"scene sampling for {len(records)} scenes", sampling_start)

    if not records:
        frame_store.close()
        raise RuntimeError("No valid scenes found for embedding extraction")

    embeds = extract_embeddings_batched(scene_clip_frames)
    filtered_records = []
    for rec, emb in zip(records, embeds):
        if emb is None:
            continue
        rec["embedding"] = emb
        filtered_records.append(rec)
    records = filtered_records

    if not records:
        frame_store.close()
        raise RuntimeError("No valid embeddings after CLIP batching")

    compute_importance(records)
    if ENABLE_MULTIMODAL_PRIOR:
        for rec in records:
            enrich_record_with_text_face(rec, whisper_words, script_scene_map)
        if identity_map.get("enabled"):
            for rec in records:
                attach_identity_links_to_record(rec, identity_map)
        apply_multimodal_boundary_priors(records)

    selected_positions, selection_tags = select_scene_positions(records)

    # Trim the selected scenes to a reasonable cap to avoid excessive VLM calls
    orig_selected_count = len(selected_positions)
    if ENABLE_LLM and len(selected_positions) > VLM_MAX_SCENES:
        # choose top scenes by importance score
        ranked = sorted(selected_positions, key=lambda i: records[i]["importance_score"], reverse=True)
        selected_positions = ranked[:VLM_MAX_SCENES]
        print(f"[VLM] Trimming selected scenes {orig_selected_count} -> {len(selected_positions)} by importance_score")

    # Keep the selected set, but process it in scene order so inference logs and cache access stay local.
    selected_positions = sorted(selected_positions)
    selected_records = [records[i] for i in selected_positions]

    segments = contiguous_scene_segments(records)
    vlm_cache = load_vlm_cache(VLM_CACHE_PATH)

    results = []
    vlm_stage_start = time.perf_counter()
    for pos in selected_positions:
        rec = records[pos]
        scene_key = str(rec["scene_id"])
        cached_desc = vlm_cache.get(scene_key)
        if cached_desc is not None:
            scene_stage_start = time.perf_counter()
            #cached_desc = normalize_vlm_description(cached_desc, rec)
            vlm_cache[scene_key] = cached_desc
            desc = {
                "scene_id": rec["scene_id"],
                "start": rec["start"],
                "end": rec["end"],
                "description": cached_desc,
            }
            log_timing(f"scene {rec['scene_id']} VLM cache hit", scene_stage_start)
        else:
            scene_stage_start = time.perf_counter()
            frame_start = time.perf_counter()
            frames = frame_store.sample_scene_frames(rec["start"], rec["end"], MAX_FRAMES_PER_SCENE)
            log_timing(f"scene {rec['scene_id']} frame sampling", frame_start)
            infer_start = time.perf_counter()
            SCENE_CONTEXT.clear()
            #SCENE_CONTEXT.update(
            #    {
            #        "transcript_text": rec.get("transcript_text", ""),
            #        "speakers": rec.get("speakers", []),
            #        "face_clusters": rec.get("face_clusters", []),
            #        "unified_character_ids": rec.get("unified_character_ids", []),
            #        "script_hint": rec.get("script_hint", ""),
            #    }
            #)
            desc = describe_scene(frames, rec["scene_id"], rec["start"], rec["end"])
            #desc["description"] = normalize_vlm_description(desc.get("description", {}), rec)
            log_timing(f"scene {rec['scene_id']} VLM inference", infer_start)
            vlm_cache[scene_key] = desc["description"]
            log_timing(f"scene {rec['scene_id']} total VLM path", scene_stage_start)

        desc["importance_score"] = rec["importance_score"]
        desc["unified_score"] = rec["unified_score"]
        desc["text_boundary_prior"] = rec.get("text_boundary_prior", 0.0)
        desc["face_boundary_prior"] = rec.get("face_boundary_prior", 0.0)
        desc["speakers"] = rec.get("speakers", [])
        desc["face_clusters"] = rec.get("face_clusters", [])
        desc["speaker_character_ids"] = rec.get("speaker_character_ids", [])
        desc["face_character_ids"] = rec.get("face_character_ids", [])
        desc["unified_character_ids"] = rec.get("unified_character_ids", [])
        desc["transcript_text"] = rec.get("transcript_text", "")
        desc["script_hint"] = rec.get("script_hint", "")
        desc["score_rank"] = rec.get("score_rank")
        desc["selection_tags"] = sorted(selection_tags[pos])
        results.append(desc)

        print(
            f"[Selected Scene {rec['scene_id']}] {rec['start']:.2f}-{rec['end']:.2f} "
            f"score={rec['importance_score']:.4f} tags={sorted(selection_tags[pos])}"
        )

    log_timing(f"VLM stage for {len(selected_positions)} scenes", vlm_stage_start)

    score_dump = []
    selected_set = set(selected_positions)
    for i, rec in enumerate(records):
        score_dump.append(
            {
                "position": i,
                "scene_id": rec["scene_id"],
                "start": rec["start"],
                "end": rec["end"],
                "importance_score": rec["importance_score"],
                "unified_score": rec["unified_score"],
                "text_boundary_prior": rec.get("text_boundary_prior", 0.0),
                "face_boundary_prior": rec.get("face_boundary_prior", 0.0),
                "score_rank": rec.get("score_rank"),
                "max_similarity_to_memory": rec["max_similarity_to_memory"],
                "selected": i in selected_set,
                "selection_tags": sorted(selection_tags.get(i, [])),
            }
        )

    summary = {
        "total_scenes": len(records),
        "selected_scenes": len(selected_records),
        "boundary_count": BOUNDARY_COUNT,
        "top_k_select": TOP_K_SELECT,
        "cluster_similarity_threshold": CLUSTER_SIM_THRESHOLD,
        "estimated_narrative_segments": len(segments),
        "segments": segments,
        "identity_reconciliation": identity_summary_payload(identity_map),
    }

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(SCORES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(score_dump, f, ensure_ascii=False, indent=2)

    with open(SELECTION_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(IDENTITY_RECONCILIATION_PATH, "w", encoding="utf-8") as f:
        json.dump(identity_summary_payload(identity_map), f, ensure_ascii=False, indent=2)

    save_vlm_cache(VLM_CACHE_PATH, vlm_cache)
    frame_store.close()

    log_timing("entire pipeline", total_start)

    print("DONE:")
    print(f"  total scenes: {len(records)}")
    print(f"  selected for VLM: {len(selected_records)}")
    print(f"  estimated segments: {len(segments)}")
    print(f"  saved: {RESULTS_JSON_PATH}, {SCORES_JSON_PATH}, {SELECTION_SUMMARY_PATH}, {IDENTITY_RECONCILIATION_PATH}, {VLM_CACHE_PATH}")


if __name__ == "__main__":
    main()