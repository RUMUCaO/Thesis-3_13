import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoProcessor, CLIPVisionModel, LlavaForConditionalGeneration
from PIL import Image
from collections import deque


# ----------------------------
# Parameter Configuration
# ----------------------------
VIDEO_PATH = "500D_clip_019.mp4"
SHOT_FRAMES = 16          # Number of frames per shot
MEMORY_SIZE = 50          # Long-term memory size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Acceleration Strategy Configuration
# ----------------------------
SAMPLE_FPS = 1.0                    # Sample only a few frames per second
MAX_VISION_FRAMES_PER_SHOT = 4      # Max frames sent to vision encoder per shot
VISION_INPUT_SIZE = 224             # Input resolution for vision model (lower = faster)
ENABLE_MIXED_PRECISION = True       # Enable mixed precision on GPU
ANALYZE_EVERY_N_SHOTS = 4           # Call LLM every N shots
KEY_SHOT_SIM_THRESHOLD = 0.92       # If similarity with previous shot is below threshold → key shot
MAX_KEYFRAMES_FOR_LLM = 1           # Max keyframes per LLM call
SIM_ON_GPU = True                   # Compute similarity matrix on GPU if possible

# Multimodal LLM Configuration
ENABLE_MULTIMODAL_ANALYSIS = True
# Optional models:
#  - "llava-hf/llava-1.5-7b-hf" (recommended, 7B, English)
#  - "llava-hf/llava-1.5-13b-hf" (13B, better performance)
#  - "Qwen/Qwen-VL-Chat" (better for Chinese, requires extra handling)
MULTIMODAL_MODEL = "llava-hf/llava-1.5-7b-hf"
MAX_FRAMES_PER_SHOT = 3  # Max keyframes kept per shot for LLM analysis

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
        llava_model = LlavaForConditionalGeneration.from_pretrained(
            MULTIMODAL_MODEL,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
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
    ...
