import cv2
import torch
import numpy as np
import json
import re
import importlib.util
from contextlib import nullcontext
from transformers import AutoImageProcessor, AutoProcessor, CLIPVisionModel
try:
    from transformers import LlavaForConditionalGeneration
except Exception:
    LlavaForConditionalGeneration = None

try:
    from transformers import Qwen2VLForConditionalGeneration
except Exception:
    Qwen2VLForConditionalGeneration = None
from PIL import Image
from collections import deque

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None


def has_accelerate():
    return importlib.util.find_spec("accelerate") is not None


# ----------------------------
# Parameter Configuration
# ----------------------------
VIDEO_PATH = "500D.mp4"
SHOT_FRAMES = 32          # Number of frames per shot
MEMORY_SIZE = 50          # Long-term memory size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Acceleration Strategy Configuration
# ----------------------------
SAMPLE_FPS = 1.0                    # Sample only a few frames per second
MAX_VISION_FRAMES_PER_SHOT = 2      # Max frames sent to vision encoder per shot
VISION_INPUT_SIZE = 224             # Input resolution for vision model (lower = faster)
ENABLE_MIXED_PRECISION = True       # Enable mixed precision on GPU
ANALYZE_EVERY_N_SHOTS = 12           # Call LLM every N shots
KEY_SHOT_SIM_THRESHOLD = 0.93       # If similarity with previous shot is below threshold → key shot
MAX_KEYFRAMES_FOR_LLM = 1           # Max keyframes per LLM call
SIM_ON_GPU = True                   # Compute similarity matrix on GPU if possible
FORCE_ANALYZE_EVERY_SHOT = False    # True = analyze every shot, no skip

# Multimodal LLM Configuration
ENABLE_MULTIMODAL_ANALYSIS = True
# Optional models:
#  - "llava-hf/llava-1.5-7b-hf" (recommended, 7B, English)
#  - "llava-hf/llava-1.5-13b-hf" (13B, better performance)
#  - "Qwen/Qwen-VL-Chat" (better for Chinese, requires extra handling)
MULTIMODAL_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
MAX_FRAMES_PER_SHOT = 1  # Max keyframes kept per shot for LLM analysis
RESULTS_JSON_PATH = "analysis_results.json"

# LLaVA generation speed settings
LLM_MAX_NEW_TOKENS = 192
LLM_MIN_NEW_TOKENS = 80
LLM_NUM_BEAMS = 1
LLM_DO_SAMPLE = False
LLM_REPETITION_PENALTY = 1.08
ENABLE_DETAIL_RETRY = True

# Quantization settings (effective on CUDA only)
ENABLE_LLM_QUANTIZATION = True
LLM_QUANTIZATION_BITS = 4  # supported: 4 or 8

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# ----------------------------
# Initialize Video Reader
# ----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ----------------------------
# Initialize Vision Feature Extractor (CLIP-Vision)
# ----------------------------
# Can be replaced with VideoMAE, TimeSformer, CLIP-ViT-L
image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
vision_model.eval()


# ----------------------------
# Transformer / Memory Module
# ----------------------------
class TemporalMemoryTransformer(torch.nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, num_layers=2):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=False,
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.memory = deque(maxlen=MEMORY_SIZE)  # Long-term memory: stores shot embeddings
        self.memory_meta = deque(maxlen=MEMORY_SIZE)

    def forward(self, x_seq):
        """
        x_seq: (seq_len, batch=1, feature_dim)
        """
        out = self.transformer(x_seq)
        return out

    def update_memory(self, scene_id, shot_embedding, keyframes=None):
        # shot_embedding: (feature_dim,) on CPU
        # keyframes: list of PIL Images
        self.memory.append(shot_embedding)
        meta = {"scene_id": scene_id, "keyframes": keyframes or []}
        self.memory_meta.append(meta)

    def retrieve_related_memories(self, query_embedding, topk=3):
        """
        query_embedding: (feature_dim,) on CPU
        Returns: [(scene_id, similarity), ...]
        """
        if len(self.memory) == 0:
            return []

        mem = torch.stack(list(self.memory), dim=0)  # (M, D)
        q = query_embedding.unsqueeze(0)             # (1, D)
        sims = torch.nn.functional.cosine_similarity(mem, q, dim=1)  # (M,)

        k = min(topk, sims.shape[0])
        values, indices = torch.topk(sims, k=k, largest=True)
        retrieved = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            retrieved.append((self.memory_meta[idx]["scene_id"], float(score)))
        return retrieved


# Initialize global temporal memory model
temporal_model = TemporalMemoryTransformer(input_dim=vision_model.config.hidden_size).to(DEVICE)

# ----------------------------
# Initialize Multimodal LLM
# ----------------------------
if ENABLE_MULTIMODAL_ANALYSIS:
    try:
        print(f"Loading multimodal LLM: {MULTIMODAL_MODEL}")
        llava_processor = AutoProcessor.from_pretrained(MULTIMODAL_MODEL)
        model_name_lower = MULTIMODAL_MODEL.lower()
        is_qwen2_vl = "qwen2-vl" in model_name_lower or "qwen/qwen2-vl" in model_name_lower
        quantization_config = None
        use_quantization = (
            DEVICE == "cuda"
            and ENABLE_LLM_QUANTIZATION
            and BitsAndBytesConfig is not None
            and LLM_QUANTIZATION_BITS in (4, 8)
            and has_accelerate()
        )

        if DEVICE == "cuda" and ENABLE_LLM_QUANTIZATION and not has_accelerate():
            print("Quantization disabled because accelerate is not installed.")

        if use_quantization:
            if LLM_QUANTIZATION_BITS == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            else:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        model_cls = Qwen2VLForConditionalGeneration if is_qwen2_vl else LlavaForConditionalGeneration
        if model_cls is None:
            raise RuntimeError(f"No compatible model class found for {MULTIMODAL_MODEL}")

        if quantization_config is not None:
            try:
                llava_model = model_cls.from_pretrained(
                    MULTIMODAL_MODEL,
                    quantization_config=quantization_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                print(f"Loaded with {LLM_QUANTIZATION_BITS}-bit quantization.")
            except Exception as quant_e:
                print(f"Quantized load failed, fallback to fp16/fp32: {quant_e}")
                llava_model = model_cls.from_pretrained(
                    MULTIMODAL_MODEL,
                    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                ).to(DEVICE)
        else:
            llava_model = model_cls.from_pretrained(
                MULTIMODAL_MODEL,
                dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            ).to(DEVICE)

        llava_model.eval()
        print("Multimodal LLM loaded successfully.")
    except Exception as e:
        print(f"Failed to load multimodal LLM: {e}")
        ENABLE_MULTIMODAL_ANALYSIS = False


# ----------------------------
# Video Frame Processing Functions
# ----------------------------
def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (VISION_INPUT_SIZE, VISION_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    processed = image_processor(images=frame_rgb, return_tensors="pt")
    return {k: v.to(DEVICE) for k, v in processed.items()}


def preprocess_frames_batch(frames):
    frame_rgbs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    frame_rgbs = [cv2.resize(f, (VISION_INPUT_SIZE, VISION_INPUT_SIZE), interpolation=cv2.INTER_AREA) for f in frame_rgbs]
    processed = image_processor(images=frame_rgbs, return_tensors="pt")
    return {k: v.to(DEVICE) for k, v in processed.items()}


def select_sampled_frames(frames, fps_value, target_fps=SAMPLE_FPS, max_frames=MAX_VISION_FRAMES_PER_SHOT):
    if not frames:
        return []

    if fps_value is None or fps_value <= 0:
        stride = max(1, len(frames) // max(1, max_frames))
    else:
        stride = max(1, int(round(fps_value / max(1e-6, target_fps))))

    sampled = frames[::stride]
    if len(sampled) > max_frames:
        idx = np.linspace(0, len(sampled) - 1, max_frames, dtype=int)
        sampled = [sampled[i] for i in idx]

    if len(sampled) == 0:
        sampled = [frames[len(frames) // 2]]
    return sampled


def llm_should_analyze(scene_id, shot_embedding, prev_shot_embedding):
    if FORCE_ANALYZE_EVERY_SHOT:
        return True

    # Rule 1: Analyze every N shots
    by_interval = (scene_id % ANALYZE_EVERY_N_SHOTS == 0)

    # Rule 2: Trigger on key shots (significant change from previous shot)
    by_key_shot = False
    if prev_shot_embedding is not None:
        sim = torch.nn.functional.cosine_similarity(
            shot_embedding.unsqueeze(0), prev_shot_embedding.unsqueeze(0), dim=1
        ).item()
        by_key_shot = sim < KEY_SHOT_SIM_THRESHOLD

    return by_interval or by_key_shot


def build_llava_generate_kwargs():
    # Balanced generation settings for stable yet detailed outputs.
    return {
        "max_new_tokens": LLM_MAX_NEW_TOKENS,
        "min_new_tokens": LLM_MIN_NEW_TOKENS,
        "num_beams": LLM_NUM_BEAMS,
        "do_sample": LLM_DO_SAMPLE,
        "repetition_penalty": LLM_REPETITION_PENALTY,
    }


def bgr_to_pil(frame_bgr):
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def extract_shot_embedding(sampled_frames):
    if not sampled_frames:
        return None

    inputs = preprocess_frames_batch(sampled_frames)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if DEVICE == "cuda" and ENABLE_MIXED_PRECISION
        else nullcontext()
    )
    with torch.no_grad():
        with autocast_ctx:
            feats = vision_model(**inputs).last_hidden_state[:, 0, :]  # (N, D)
            shot_embedding = feats.mean(dim=0)  # (D,)

    return torch.nn.functional.normalize(shot_embedding, dim=0).detach().cpu()


def parse_llm_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def extract_partial_fields(text):
    # Fallback parser for truncated JSON-like responses.
    patterns = {
        "description": r'"description"\s*:\s*"([^\"]*)',
        "dynamics": r'"dynamics"\s*:\s*"([^\"]*)',
        "relation_state": r'"relation_state"\s*:\s*"([^\"]*)',
    }
    extracted = {"description": "", "dynamics": "", "relation_state": ""}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            extracted[key] = match.group(1).replace("\\n", " ").strip()

    if any(extracted.values()):
        return extracted
    return None


def is_low_detail_result(description, dynamics, relation_state):
    desc_words = len(description.split())
    has_relation = bool(relation_state and relation_state.lower() not in ("unknown", "uncertain"))
    has_dynamics = bool(dynamics)
    return desc_words < 16 or (not has_relation and not has_dynamics)


def run_llava_json_inference(prompt, image_for_llm):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    prompt_with_image_token = llava_processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
    )
    llava_inputs = llava_processor(
        text=prompt_with_image_token,
        images=image_for_llm,
        return_tensors="pt",
    )

    model_device = next(llava_model.parameters()).device
    llava_inputs = {
        k: (v.to(model_device) if torch.is_tensor(v) else v)
        for k, v in llava_inputs.items()
    }

    with torch.no_grad():
        output_ids = llava_model.generate(
            **llava_inputs,
            **build_llava_generate_kwargs(),
        )

    prompt_tokens = llava_inputs["input_ids"].shape[1]
    new_token_ids = output_ids[:, prompt_tokens:]
    output_text = llava_processor.batch_decode(new_token_ids, skip_special_tokens=True)[0].strip()
    parsed = parse_llm_json(output_text)
    if parsed is None:
        parsed = extract_partial_fields(output_text)
    if parsed is None:
        parsed = {
            "description": output_text.splitlines()[0][:320].strip(),
            "dynamics": "",
            "relation_state": "",
        }

    return {
        "description": str(parsed.get("description", "")).strip(),
        "dynamics": str(parsed.get("dynamics", "")).strip(),
        "relation_state": str(parsed.get("relation_state", "")).strip(),
    }


def analyze_shot_with_llm(keyframes, shot_embedding, prev_analysis, scene_id):
    """
    Analyze keyframes of a shot using multimodal LLM
    keyframes: list of PIL Images
    shot_embedding: (D,) CPU tensor
    prev_analysis: previous shot analysis (dict) or None
    
    Returns:
    {
        "scene_id": int,
        "description": str,
        "dynamics": str,
        "relation_state": str
    }
    """
    default_result = {
        "scene_id": scene_id,
        "description": "",
        "dynamics": "",
        "relation_state": "",
    }

    if not ENABLE_MULTIMODAL_ANALYSIS or len(keyframes) == 0:
        return default_result

    try:
        related = temporal_model.retrieve_related_memories(shot_embedding, topk=2)
        memory_text = ", ".join([f"scene {sid} ({score:.2f})" for sid, score in related]) or "none"
        prev_summary = prev_analysis["description"] if prev_analysis else "none"

        prompt = (
            "You are analyzing a movie shot. "
            "Return ONLY valid JSON with exactly keys: description, dynamics, relation_state. "
            "description: 2-3 sentences with concrete visual details: characters, clothing, objects, setting, and interaction cues. "
            "dynamics: describe what is changing in this shot relative to surrounding context. "
            "relation_state: infer interpersonal state (e.g., tense/cooperative/distant/uncertain) and include a short reason. "
            "No markdown, no extra keys, no explanations. "
            f"Current scene id: {scene_id}. "
            f"Previous scene summary: {prev_summary}. "
            f"Related memory scenes: {memory_text}."
        )

        image_for_llm = keyframes[0]
        parsed = run_llava_json_inference(prompt, image_for_llm)

        description = parsed["description"]
        dynamics = parsed["dynamics"]
        relation_state = parsed["relation_state"]

        if ENABLE_DETAIL_RETRY and is_low_detail_result(description, dynamics, relation_state):
            retry_prompt = (
                "Refine the same shot with higher granularity. "
                "Return ONLY valid JSON with keys: description, dynamics, relation_state. "
                "Add specific visual evidence: body posture, gaze direction, props, spatial arrangement, and implied social tension. "
                "Do not repeat generic phrases."
            )
            refined = run_llava_json_inference(retry_prompt, image_for_llm)
            if len(refined["description"].split()) > len(description.split()):
                description = refined["description"] or description
                dynamics = refined["dynamics"] or dynamics
                relation_state = refined["relation_state"] or relation_state

        if not relation_state:
            relation_state = "uncertain"

        return {
            "scene_id": scene_id,
            "description": description,
            "dynamics": dynamics,
            "relation_state": relation_state,
        }
    except Exception as e:
        return {
            "scene_id": scene_id,
            "description": f"LLM analysis failed: {e}",
            "dynamics": "",
            "relation_state": "",
        }


def process_one_shot(scene_id, shot_frames, prev_shot_embedding, prev_analysis):
    sampled = select_sampled_frames(
        shot_frames,
        fps_value=fps,
        target_fps=SAMPLE_FPS,
        max_frames=MAX_VISION_FRAMES_PER_SHOT,
    )
    if not sampled:
        scene_result = {
            "scene_id": scene_id,
            "status": "skipped",
            "description": "",
            "dynamics": "",
            "relation_state": "",
            "reason": "no_sampled_frames",
        }
        return prev_shot_embedding, prev_analysis, scene_result

    shot_embedding = extract_shot_embedding(sampled)
    if shot_embedding is None:
        scene_result = {
            "scene_id": scene_id,
            "status": "skipped",
            "description": "",
            "dynamics": "",
            "relation_state": "",
            "reason": "embedding_failed",
        }
        return prev_shot_embedding, prev_analysis, scene_result

    keyframes = [bgr_to_pil(f) for f in sampled[:max(1, MAX_FRAMES_PER_SHOT)]]
    temporal_model.update_memory(scene_id, shot_embedding, keyframes=keyframes)

    should_analyze = llm_should_analyze(scene_id, shot_embedding, prev_shot_embedding)
    if should_analyze and ENABLE_MULTIMODAL_ANALYSIS:
        analysis = analyze_shot_with_llm(keyframes, shot_embedding, prev_analysis, scene_id)
        status = "failed" if str(analysis.get("description", "")).startswith("LLM analysis failed:") else "analyzed"
        scene_result = {
            "scene_id": scene_id,
            "status": status,
            "description": analysis.get("description", ""),
            "dynamics": analysis.get("dynamics", ""),
            "relation_state": analysis.get("relation_state", ""),
            "reason": "",
        }
        print(
            f"[Scene {scene_id}] description={analysis['description']} | "
            f"dynamics={analysis['dynamics']} | relation_state={analysis['relation_state']}"
        )
        prev_analysis = analysis
    else:
        scene_result = {
            "scene_id": scene_id,
            "status": "skipped",
            "description": "",
            "dynamics": "",
            "relation_state": "",
            "reason": "policy_skip",
        }
        print(f"[Scene {scene_id}] skipped LLM analysis.")

    return shot_embedding, prev_analysis, scene_result


def run_video_pipeline():
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    print(
        f"Start processing: {VIDEO_PATH} | fps={fps:.2f} | "
        f"frames={frame_count} | resolution={frame_width}x{frame_height}"
    )

    scene_id = 0
    current_shot_frames = []
    prev_shot_embedding = None
    prev_analysis = None
    all_scene_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_shot_frames.append(frame)
        if len(current_shot_frames) >= SHOT_FRAMES:
            scene_id += 1
            prev_shot_embedding, prev_analysis, scene_result = process_one_shot(
                scene_id,
                current_shot_frames,
                prev_shot_embedding,
                prev_analysis,
            )
            all_scene_results.append(scene_result)
            current_shot_frames = []

    if current_shot_frames:
        scene_id += 1
        prev_shot_embedding, prev_analysis, scene_result = process_one_shot(
            scene_id,
            current_shot_frames,
            prev_shot_embedding,
            prev_analysis,
        )
        all_scene_results.append(scene_result)

    cap.release()
    analyzed_count = sum(1 for r in all_scene_results if r["status"] == "analyzed")
    failed_count = sum(1 for r in all_scene_results if r["status"] == "failed")
    skipped_count = sum(1 for r in all_scene_results if r["status"] == "skipped")

    results_payload = {
        "video_path": VIDEO_PATH,
        "fps": float(fps),
        "frame_count": int(frame_count),
        "resolution": {"width": int(frame_width), "height": int(frame_height)},
        "total_scenes": int(scene_id),
        "summary": {
            "analyzed": int(analyzed_count),
            "failed": int(failed_count),
            "skipped": int(skipped_count),
        },
        "scenes": all_scene_results,
    }
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)

    print(f"Video analysis finished. total_scenes={scene_id}")
    print(
        "Saved results to "
        f"{RESULTS_JSON_PATH} (analyzed={analyzed_count}, failed={failed_count}, skipped={skipped_count})"
    )


if __name__ == "__main__":
    run_video_pipeline()
