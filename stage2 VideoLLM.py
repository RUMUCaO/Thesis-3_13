from __future__ import annotations

import cv2
import torch
import numpy as np
import json
import re
import importlib.util
import time
from collections import OrderedDict, deque
from PIL import Image

from transformers import CLIPVisionModel

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
VIDEO_PATH = "output_0.5x.mp4"
SCENE_JSON_PATH = "semantic_scenes.json"
RESULTS_JSON_PATH = "scene_level_results.json"
SCORES_JSON_PATH = "scene_scores.json"
SELECTION_SUMMARY_PATH = "scene_selection_summary.json"
VLM_CACHE_PATH = "vlm_cache.json"

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
        end_i = int(end * self.fps)
        end_i = max(start_i + 1, end_i)

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

    if isinstance(data, dict) and "scenes" in data:
        return data["scenes"]
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
        key=lambda i: records[i]["importance_score"],
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
    scenes = load_scenes(SCENE_JSON_PATH)
    frame_store = VideoFrameStore(VIDEO_PATH)

    records = []
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
            desc = describe_scene(frames, rec["scene_id"], rec["start"], rec["end"])
            log_timing(f"scene {rec['scene_id']} VLM inference", infer_start)
            vlm_cache[scene_key] = desc["description"]
            log_timing(f"scene {rec['scene_id']} total VLM path", scene_stage_start)

        desc["importance_score"] = rec["importance_score"]
        desc["unified_score"] = rec["unified_score"]
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
    }

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(SCORES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(score_dump, f, ensure_ascii=False, indent=2)

    with open(SELECTION_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    save_vlm_cache(VLM_CACHE_PATH, vlm_cache)
    frame_store.close()

    log_timing("entire pipeline", total_start)

    print("DONE:")
    print(f"  total scenes: {len(records)}")
    print(f"  selected for VLM: {len(selected_records)}")
    print(f"  estimated segments: {len(segments)}")
    print(f"  saved: {RESULTS_JSON_PATH}, {SCORES_JSON_PATH}, {SELECTION_SUMMARY_PATH}, {VLM_CACHE_PATH}")


if __name__ == "__main__":
    main()