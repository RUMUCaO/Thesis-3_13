import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoProcessor, CLIPVisionModel, LlavaForConditionalGeneration
from PIL import Image
from collections import deque


# ----------------------------
# 参数配置
# ----------------------------
VIDEO_PATH = "500D_clip_019.mp4"
SHOT_FRAMES = 16          # 每个镜头抽取帧数
MEMORY_SIZE = 50          # 长时记忆长度
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# 加速策略配置
# ----------------------------
SAMPLE_FPS = 1.0                    # 每秒仅抽取少量帧
MAX_VISION_FRAMES_PER_SHOT = 4      # 每个镜头最多送入视觉编码器的帧数
VISION_INPUT_SIZE = 224             # 视觉输入分辨率（降低可提速）
ENABLE_MIXED_PRECISION = True       # GPU 混合精度
ANALYZE_EVERY_N_SHOTS = 4           # 每 N 个镜头调用一次 LLM
KEY_SHOT_SIM_THRESHOLD = 0.92       # 与上一镜头相似度低于阈值，视为关键镜头
MAX_KEYFRAMES_FOR_LLM = 1           # 每次 LLM 最多分析关键帧数
SIM_ON_GPU = True                   # 相似度矩阵尽量放 GPU

# Multimodal LLM 配置
ENABLE_MULTIMODAL_ANALYSIS = True
# 可选模型：
#  - "llava-hf/llava-1.5-7b-hf" (推荐，7B，英文)
#  - "llava-hf/llava-1.5-13b-hf" (13B，更好)
#  - "Qwen/Qwen-VL-Chat" (中文友好，但需要特殊处理)
MULTIMODAL_MODEL = "llava-hf/llava-1.5-7b-hf"
MAX_FRAMES_PER_SHOT = 3  # 每个镜头保留最多 3 个关键帧供 LLM 分析

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# ----------------------------
# 初始化视频读取
# ----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ----------------------------
# 初始化视觉特征提取器（CLIP-Vision）
# ----------------------------
# 可以换成 VideoMAE, TimeSformer, CLIP-ViT-L
image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
vision_model.eval()


# ----------------------------
# Transformer / Memory
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
        self.memory = deque(maxlen=MEMORY_SIZE)  # 长时记忆: 存每个镜头向量
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
        返回: [(scene_id, similarity), ...]
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


# 初始化全局记忆 Transformer
temporal_model = TemporalMemoryTransformer(input_dim=vision_model.config.hidden_size).to(DEVICE)

# ----------------------------
# 初始化 Multimodal LLM
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
# 视频帧处理函数
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
    # 规则1：每 N 个镜头分析一次
    by_interval = (scene_id % ANALYZE_EVERY_N_SHOTS == 0)

    # 规则2：关键镜头（与上一镜头差异明显）额外触发分析
    by_key_shot = False
    if prev_shot_embedding is not None:
        sim = torch.nn.functional.cosine_similarity(
            shot_embedding.unsqueeze(0), prev_shot_embedding.unsqueeze(0), dim=1
        ).item()
        by_key_shot = sim < KEY_SHOT_SIM_THRESHOLD

    return by_interval or by_key_shot


def analyze_shot_with_llm(keyframes, shot_embedding, prev_analysis, scene_id):
    """
    用 multimodal LLM 分析一个镜头的关键帧
    keyframes: list of PIL Images
    shot_embedding: (D,) CPU tensor
    prev_analysis: 前一个镜头的分析结果 (dict) 或 None
    
    返回: {"scene_id": int, "description": str, "dynamics": str, "relation_state": str}
    """
    if not ENABLE_MULTIMODAL_ANALYSIS:
        return {
            "scene_id": scene_id,
            "description": "Multimodal analysis disabled.",
            "dynamics": "N/A",
            "relation_state": "N/A",
        }

    try:
        if not keyframes:
            return {"scene_id": scene_id, "description": "No keyframes.", "dynamics": "N/A", "relation_state": "N/A"}

        # 使用少量关键帧，批量调用 LLaVA
        keyframes = keyframes[:MAX_KEYFRAMES_FOR_LLM]

        # 构造多模态 prompt
        user_text = (
            "You are a film analyst. Analyze this shot from a video. "
            "Describe: 1) Scene content and setting, 2) Character dynamics and actions, "
            "3) Emotional tone or relationship state. Be concise and specific."
        )
        
        if prev_analysis:
            user_text += f" Previous shot analysis: {prev_analysis.get('description', 'N/A')}"

        # LLaVA-1.5 常用对话模板（每张图一个 prompt）
        prompt_texts = [f"USER: <image>\n{user_text}\nASSISTANT:" for _ in keyframes]

        # LLaVA 批量推理
        inputs = llava_processor(images=keyframes, text=prompt_texts, return_tensors="pt", padding=True).to(DEVICE)
        with torch.inference_mode():
            output_ids = llava_model.generate(
                **inputs,
                max_new_tokens=120,
                temperature=0.7,
                top_p=0.9,
                do_sample=False,
            )

        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        decoded = []
        for i in range(output_ids.shape[0]):
            gen_ids = output_ids[i, int(input_lens[i]):]
            text_i = llava_processor.decode(gen_ids, skip_special_tokens=True).strip()
            if text_i:
                decoded.append(text_i)

        analysis_text = " ".join(decoded) if decoded else "No textual response from LLM."

        # 简单的后处理：提取关系状态
        relation_state = "持续发展"
        if "conflict" in analysis_text.lower() or "change" in analysis_text.lower():
            relation_state = "关系变化"
        elif "same" in analysis_text.lower() or "continue" in analysis_text.lower():
            relation_state = "关系延续"

        return {
            "scene_id": scene_id,
            "description": analysis_text,
            "dynamics": analysis_text,
            "relation_state": relation_state,
        }
    except Exception as e:
        return {
            "scene_id": scene_id,
            "description": f"LLM analysis failed: {e}",
            "dynamics": "Error",
            "relation_state": "Error",
        }


def extract_keyframes(frames, num_frames=MAX_FRAMES_PER_SHOT):
    """
    从一个镜头的帧列表中均匀抽取关键帧
    """
    if not frames:
        return []
    if len(frames) <= num_frames:
        keyframes = frames
    else:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        keyframes = [frames[i] for i in indices]
    
    # 转换为 PIL Image
    pil_frames = []
    for f in keyframes:
        if isinstance(f, np.ndarray):
            # OpenCV 格式 (BGR) -> RGB
            f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(f_rgb))
        else:
            pil_frames.append(f)
    return pil_frames


def print_global_summary(scene_records):
    total_scenes = len(scene_records)
    if total_scenes == 0:
        print("未检测到有效镜头，无法做全局分析。")
        return

    print("==== Global Scene Analysis ====")
    print(f"视频信息: fps={fps:.2f}, frames={frame_count}, resolution={frame_width}x{frame_height}")
    print(f"总镜头数: {total_scenes}")

    print("\n==== 逐镜头 Multimodal LLM 分析 ====")
    for r in scene_records:
        print(f"\nScene {r['scene_id']:03d}:")
        print(f"  Description: {r.get('description', 'N/A')[:200]}...")
        print(f"  Relation State: {r.get('relation_state', 'N/A')}")

    # 计数关系状态
    relation_counts = {}
    for r in scene_records:
        state = r.get("relation_state", "Unknown")
        relation_counts[state] = relation_counts.get(state, 0) + 1

    print("\n==== 关系状态统计 ====")
    for state, count in relation_counts.items():
        print(f"{state}: {count} scenes")

    # 全局总结（可选：再用一次 LLM）
    print("\n==== 全局电影叙事总结 ====")
    try:
        if ENABLE_MULTIMODAL_ANALYSIS and len(scene_records) > 0:
            all_descriptions = "\n".join(
                [f"Scene {r['scene_id']}: {r.get('description', 'N/A')[:100]}" for r in scene_records[:5]]
            )
            
            summary_prompt = (
                f"Given the following shot-level analyses from a video, "
                f"summarize the overall narrative arc, character development, and relationship dynamics:\n\n{all_descriptions}\n\n"
                "Provide a concise summary."
            )
            
            # 这里也可以用文本 LLM，不一定要用 multimodal
            # 但我们简化处理：直接输出前几个镜头的分析
            print(f"[Based on {len(scene_records)} shot analyses]")
            for r in scene_records[:3]:
                print(f"  - Scene {r['scene_id']}: {r.get('description', 'N/A')[:150]}")
    except Exception as e:
        print(f"Global summary generation failed: {e}")


def process_shot(frames, scene_id, prev_analysis, shot_embeddings, scene_records):
    # 1) 帧采样 + 批量视觉特征（加速）
    sampled_frames = select_sampled_frames(frames, fps)
    if len(sampled_frames) == 0:
        sampled_frames = [frames[0]]

    vision_inputs = preprocess_frames_batch(sampled_frames)
    with torch.inference_mode():
        if DEVICE == "cuda" and ENABLE_MIXED_PRECISION:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                frame_feats = vision_model(**vision_inputs).last_hidden_state.mean(dim=1)
        else:
            frame_feats = vision_model(**vision_inputs).last_hidden_state.mean(dim=1)

    frame_feats = frame_feats.float()                        # (T, D)
    frame_feats_seq = frame_feats.unsqueeze(1)               # (T, 1, D)

    # 2) 时序 Transformer
    with torch.inference_mode():
        if DEVICE == "cuda" and ENABLE_MIXED_PRECISION:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out_seq = temporal_model(frame_feats_seq)
        else:
            out_seq = temporal_model(frame_feats_seq)

    # 3) 镜头 embedding
    shot_embedding = out_seq.mean(dim=0).squeeze(0).detach().cpu()  # (D,)

    # 4) 关键帧提取
    keyframes = extract_keyframes(sampled_frames, num_frames=MAX_FRAMES_PER_SHOT)

    # 5) 减少 LLM 调用频率：只分析间隔镜头或关键镜头
    prev_shot_embedding = shot_embeddings[-1].squeeze(0) if len(shot_embeddings) > 0 else None
    should_analyze = llm_should_analyze(scene_id, shot_embedding, prev_shot_embedding)
    if should_analyze:
        analysis = analyze_shot_with_llm(keyframes, shot_embedding, prev_analysis, scene_id)
    else:
        analysis = {
            "scene_id": scene_id,
            "description": "Skipped LLM for speed (non-key shot).",
            "dynamics": "N/A",
            "relation_state": "未分析",
        }

    shot_embeddings.append(shot_embedding.unsqueeze(0))
    scene_records.append(analysis)

    # 6) 更新长时记忆（存储 embedding + keyframes）
    temporal_model.update_memory(scene_id, shot_embedding, keyframes=keyframes)

    print(
        f"Scene {scene_id}: processed {len(sampled_frames)}/{len(frames)} sampled frames | "
        f"llm={should_analyze} | "
        f"relation_state={analysis.get('relation_state', 'N/A')} | "
        f"description={analysis.get('description', 'N/A')[:80]}..."
    )
    return analysis


# ----------------------------
# 主循环
# ----------------------------
shot_frames_buffer = []
shot_embeddings = []
scene_records = []

scene_id = 0
prev_analysis = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    shot_frames_buffer.append(frame)
    if len(shot_frames_buffer) >= SHOT_FRAMES:
        prev_analysis = process_shot(
            shot_frames_buffer,
            scene_id,
            prev_analysis,
            shot_embeddings,
            scene_records,
        )
        scene_id += 1
        shot_frames_buffer = []

cap.release()

# 处理最后不足 SHOT_FRAMES 的尾段
if len(shot_frames_buffer) > 0:
    prev_analysis = process_shot(
        shot_frames_buffer,
        scene_id,
        prev_analysis,
        shot_embeddings,
        scene_records,
    )
    scene_id += 1

# ----------------------------
# 跨镜头全局分析
# ----------------------------
if len(shot_embeddings) > 0:
    shot_embeddings_tensor = torch.cat(shot_embeddings, dim=0)  # (num_shots, D)

    # GPU 并行矩阵计算：cosine_sim = normalize(X) @ normalize(X)^T
    sim_device = "cuda" if (SIM_ON_GPU and torch.cuda.is_available()) else "cpu"
    x = shot_embeddings_tensor.to(sim_device)
    x = torch.nn.functional.normalize(x, p=2, dim=1)
    sim_matrix = torch.matmul(x, x.t()).detach().cpu()

    print("Shot-to-shot similarity matrix computed.")
    print("Temporal Memory contains", len(temporal_model.memory), "historical shots for long-term context.")
else:
    print("No shot embeddings generated.")

print_global_summary(scene_records)

# ----------------------------
# 说明
# 1) Multimodal LLM Pipeline:
#    - 每帧視覺編碼 (CLIP) -> 提取第一個鍵幀 -> LLaVA/Qwen-VL 分析
# 2) LLM 输出包含:
#    - Scene description (内容描述)
#    - Character dynamics (人物动态)
#    - Relation state (关系状态)
# 3) 长时记忆保存 embedding + keyframes，支持跨镜头对比
# 4) 保留了 shot embedding لل future 扩展 (聚类、Re-ID 等)
# ----------------------------