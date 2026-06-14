from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
import subprocess
import open_clip
import whisper
import torchaudio
import tempfile
from scenedetect import detect, ContentDetector, SceneManager, open_video

def repair_video_with_ffmpeg(input_path: Path, output_path: Path = None, crf: int = 23) -> Path:
    """
    用 FFmpeg 强制重新编码视频，修复损坏的 H.264 数据流。
    
    参数:
        input_path: 原始视频文件路径
        output_path: 修复后视频保存路径（若为 None，则自动生成临时文件）
        crf: 画质参数 (默认 23，越小画质越好但文件越大)
    
    返回:
        修复后的视频文件路径
    """
    if output_path is None:
        # 使用临时文件，避免污染原始文件
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="repaired_")
        output_path = Path(output_path)
        # 关闭文件描述符，ffmpeg 会覆盖它
        import os
        os.close(fd)
    
    # 构建 ffmpeg 命令：重新编码视频（H.264），音频（AAC），覆盖输出
    cmd = [
        "ffmpeg", "-i", str(input_path),
        "-c:v", "libx264", "-crf", str(crf), "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        "-y", str(output_path)
    ]
    print(f"正在修复视频（重新编码）: {input_path.name} -> {output_path.name}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("修复完成")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"修复失败: {e}")
        raise

# =========================
# paths
# =========================
VIDEO_PATH = Path("PW.mp4")
SCENE_JSON = Path("stage2_TransNetV2_scenes.json")
SEMANTIC_JSON = Path("semantic_scenes.json")


# =========================
# CLIP model
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
 
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
clip_model = clip_model.to(device).eval()


@torch.no_grad()
def encode_image(img):
    x = preprocess(img).unsqueeze(0).to(device)
    f = clip_model.encode_image(x)
    f = f / f.norm(dim=-1, keepdim=True)
    return f.squeeze(0).cpu().numpy()


# =========================
# Whisper (audio → text)
# =========================
whisper_model = whisper.load_model("base")

def get_scenes_with_scenedetect(video_path: Path, threshold=30):
    """
    使用 scenedetect 检测镜头边界，返回场景列表，格式与 TransNetV2 兼容。
    threshold: 内容检测灵敏度（越低越敏感）。
    """
    from scenedetect import detect, ContentDetector
    scene_list = detect(str(video_path), ContentDetector(threshold=threshold))
    scenes = []
    for i, (start_frame, end_frame) in enumerate(scene_list):
        scenes.append({
            "scene_id": i,
            "start_frame": start_frame.frame_num,
            "end_frame": end_frame.frame_num,
            "start_seconds": start_frame.get_seconds(),
            "end_seconds": end_frame.get_seconds(),
        })
    return scenes

def extract_full_audio(video_path: Path, out_wav: str = "full_audio.wav"):
    """提取整个视频的音频到 wav 文件，返回音频数组和采样率"""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", "16000", out_wav
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    audio, sr = torchaudio.load(out_wav)
    Path(out_wav).unlink()  # 删除临时文件
    return audio, sr

def get_audio_segment(full_audio, sr, start_sec, end_sec):
    """从完整音频中截取片段"""
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    return full_audio[:, start_sample:end_sample]

# =========================
# multi-frame CLIP (核心升级🔥)
# =========================
def get_frame(video_path, t):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def multi_frame_embedding(video_path, scene, n=5):
    """
    ⭐ 关键升级：scene级 embedding
    """
    start = scene["start_seconds"]
    end = scene["end_seconds"]

    ts = np.linspace(start, end, n)

    embs = []
    for t in ts:
        img = get_frame(video_path, t)
        if img is None:
            continue
        embs.append(encode_image(img))

    if not embs:
        return np.zeros(512)

    return np.mean(embs, axis=0)


# =========================
# load scenes
# =========================
def load_scenes():
    return json.loads(SCENE_JSON.read_text())["scenes"]


# =========================
# semantic merge (fusion版🔥)
# =========================
def merge_scenes(scenes, visual_embs, audio_embs, alpha=0.7, threshold=0.75):
    def sim(a, b):
        return float(np.dot(a, b))
    groups = []
    cur = [scenes[0]]
    for i in range(1, len(scenes)):
        v_sim = sim(visual_embs[i-1], visual_embs[i])
        a_sim = sim(audio_embs[i-1], audio_embs[i])
        score = alpha * v_sim + (1 - alpha) * a_sim
        if score > threshold:
            cur.append(scenes[i])
        else:
            groups.append(cur)
            cur = [scenes[i]]
    groups.append(cur)
    return groups

def to_semantic(groups):
    return [
        {
            "index": i,
            "start": g[0]["start_seconds"],
            "end": g[-1]["end_seconds"],
            "scene_count": len(g)
        }
        for i, g in enumerate(groups)
    ]

# =========================
# main
# ========================= 
def main():
    print("Detecting scenes with scenedetect...")
    repaired_video_path = repair_video_with_ffmpeg(VIDEO_PATH)# 尝试修复视频，避免解码错误导致的场景检测失败
    scenes = get_scenes_with_scenedetect(repaired_video_path, threshold=30)   # 可调
    if not scenes:
        print("No scenes detected, abort.")
        return
    
    # 可选：保存原始场景（方便复查）
    SCENE_JSON.write_text(json.dumps({"scenes": scenes}, indent=2), encoding="utf-8")

    # -------------------------
    # 1. multi-frame CLIP
    # -------------------------
    print("Building multi-frame CLIP embeddings...")
    visual_embs = []

    for s in scenes:
        emb = multi_frame_embedding(repaired_video_path, s)
        visual_embs.append(emb)

    visual_embs = np.array(visual_embs)

    # -------------------------
    # 2. audio embedding
    # -------------------------
    print("Extracting audio semantic signal...")
    full_audio, audio_sr = extract_full_audio(repaired_video_path)
    audio_embs = []

    for s in scenes:
        start = s["start_seconds"]
        end = s["end_seconds"]
        if end - start < 0.5:   # 太短的片段直接给零向量
            audio_embs.append(np.zeros(512))
            continue
        
        segment = get_audio_segment(full_audio, audio_sr, start, end)
        
        # 1. 转为单声道 (如果 segment 是多通道)
        if segment.dim() > 1 and segment.shape[0] > 1:
            segment = segment.mean(dim=0, keepdim=True).squeeze()
        else:
            segment = segment.squeeze()
        
        # 2. 转成 float32 numpy
        audio_np = segment.cpu().numpy().astype(np.float32)
        
        # 3. 重采样到 16000 Hz (如果原始采样率不是 16000)
        if audio_sr != 16000:
            import torchaudio
            # 先转回 tensor
            audio_tensor = torch.from_numpy(audio_np)
            # torchaudio 重采样
            resampler = torchaudio.transforms.Resample(orig_freq=audio_sr, new_freq=16000)
            audio_np = resampler(audio_tensor).numpy()
        
        # 4. 关键：填充/裁剪到 30 秒 (480000 个采样点)
        audio_30s = whisper.pad_or_trim(torch.from_numpy(audio_np))   # 输出 shape (480000,)
        
        # 5. 计算梅尔谱 (shape: 80, 3000)
        mel = whisper.log_mel_spectrogram(audio_30s)   # 注意：不要 unsqueeze 在这里
        
        # 6. 添加 batch 维度并送入 encoder
        mel_batch = mel.unsqueeze(0).to(device)        # shape (1, 80, 3000)
        with torch.no_grad():
            features = whisper_model.encoder(mel_batch)   # shape (1, 1500, 512)
        
        # 7. 取时间维度的平均作为片段嵌入
        emb = features.mean(dim=1).squeeze().cpu().numpy()  # shape (512,)
        audio_embs.append(emb)

    # -------------------------
    # 3. semantic merge
    # -------------------------
    print("Merging semantic scenes...")
    groups = merge_scenes(scenes, visual_embs, audio_embs, alpha=0.7, threshold=0.75)

    semantic = to_semantic(groups)

    SEMANTIC_JSON.write_text(
        json.dumps({
            "video": repaired_video_path.name,
            "semantic_scenes": semantic
        }, indent=2),
        encoding="utf-8"
    )

    print("Done")
    print("raw scenes:", len(scenes))
    print("semantic scenes:", len(semantic))


if __name__ == "__main__":
    main()