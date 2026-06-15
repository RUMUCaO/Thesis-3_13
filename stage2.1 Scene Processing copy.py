from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
import subprocess
import open_clip
import whisper
import torchaudio
from scenedetect import detect, ContentDetector
from tqdm import tqdm

# 尝试导入 decord
try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    print("警告: decord 未安装，将使用较慢的 OpenCV 逐帧读取。建议安装: pip install decord")

# ---------------------------
# 视频修复（同前）
# ---------------------------
def repair_video_with_ffmpeg(input_path: Path, output_path: Path = None, crf: int = 23) -> Path:
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="repaired_")
        output_path = Path(output_path)
        os.close(fd)
    cmd = [
        "ffmpeg", "-i", str(input_path),
        "-c:v", "libx264", "-crf", str(crf), "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        "-y", str(output_path)
    ]
    print(f"修复视频: {input_path.name} -> {output_path.name}")
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

# ---------------------------
# 路径配置
# ---------------------------
VIDEO_PATH = Path("PW.mp4")
SCENE_JSON = Path("stage2_TransNetV2_scenes.json")
SEMANTIC_JSON = Path("semantic_scenes.json")

# ---------------------------
# CLIP 模型
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
clip_model = clip_model.to(device).eval()

@torch.no_grad()
def encode_image(img):
    x = preprocess(img).unsqueeze(0).to(device)
    f = clip_model.encode_image(x)
    f = f / f.norm(dim=-1, keepdim=True)
    return f.squeeze(0).cpu().numpy()

# ---------------------------
# Whisper 模型
# ---------------------------
whisper_model = whisper.load_model("base")

# ---------------------------
# 场景检测
# ---------------------------
def get_scenes_with_scenedetect(video_path: Path, threshold=30):
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

# ---------------------------
# 音频处理
# ---------------------------
def extract_full_audio(video_path: Path, out_wav="full_audio.wav"):
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", "16000", out_wav]
    subprocess.run(cmd, check=True, capture_output=True)
    audio, sr = torchaudio.load(out_wav)
    Path(out_wav).unlink()
    return audio, sr

def get_audio_segment(full_audio, sr, start_sec, end_sec):
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    return full_audio[:, start_sample:end_sample]

# ---------------------------
# 语义合并（使用余弦相似度）
# ---------------------------
def normalized_sim(a, b):
    a_n = a / (np.linalg.norm(a) + 1e-8)
    b_n = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_n, b_n))

def merge_scenes(scenes, visual_embs, audio_embs, alpha=0.7, threshold=0.65):
    groups = []
    cur = [scenes[0]]
    for i in range(1, len(scenes)):
        v_sim = normalized_sim(visual_embs[i-1], visual_embs[i])
        a_sim = normalized_sim(audio_embs[i-1], audio_embs[i])
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

# ---------------------------
# 主函数
# ---------------------------
def main():
    print("1. 修复视频...")
    repaired_video = repair_video_with_ffmpeg(VIDEO_PATH)

    print("2. 检测原始场景...")
    scenes = get_scenes_with_scenedetect(repaired_video, threshold=30)
    if not scenes:
        print("未检测到任何场景，退出")
        return
    SCENE_JSON.write_text(json.dumps({"scenes": scenes}, indent=2))
    print(f"检测到 {len(scenes)} 个场景")

    # ---------- 准备视频读取器 ----------
    print("3. 准备视频帧提取器...")
    if DECORD_AVAILABLE:
        vr = decord.VideoReader(str(repaired_video), ctx=decord.cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        def get_frame(timestamp_sec):
            idx = int(timestamp_sec * fps)
            idx = max(0, min(idx, total_frames-1))
            img = vr[idx].asnumpy()
            return Image.fromarray(img)
    else:
        # OpenCV fallback：仍然每次打开视频效率低，但作为备选
        cap = cv2.VideoCapture(str(repaired_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        def get_frame(timestamp_sec):
            idx = int(timestamp_sec * fps)
            idx = max(0, min(idx, total_frames-1))
            cap = cv2.VideoCapture(str(repaired_video))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame)

    # ---------- 视觉特征提取 ----------
    print("4. 提取 CLIP 视觉特征...")
    visual_embs = []
    n_frames_per_scene = 3  # 减少采样帧数，提高速度
    for s in tqdm(scenes, desc="处理场景"):
        start = s["start_seconds"]
        end = s["end_seconds"]
        if end - start < 0.2:
            # 场景太短，使用零向量（后续会被合并）
            visual_embs.append(np.zeros(512))
            continue
        ts = np.linspace(start, end, n_frames_per_scene)
        embs = []
        for t in ts:
            img = get_frame(t)
            if img is not None:
                embs.append(encode_image(img))
        if len(embs) == 0:
            visual_embs.append(np.zeros(512))
        else:
            visual_embs.append(np.mean(embs, axis=0))
    visual_embs = np.array(visual_embs)

    # 调试输出
    print(f"视觉特征 shape: {visual_embs.shape}, 均值: {visual_embs.mean():.6f}, 标准差: {visual_embs.std():.6f}")
    zero_count = np.sum(np.linalg.norm(visual_embs, axis=1) < 1e-6)
    print(f"零向量数量: {zero_count}/{len(visual_embs)}")

    # ---------- 音频特征提取 ----------
    print("5. 提取 Whisper 音频特征...")
    full_audio, audio_sr = extract_full_audio(repaired_video)
    audio_embs = []
    for s in tqdm(scenes, desc="处理音频"):
        start = s["start_seconds"]
        end = s["end_seconds"]
        if end - start < 0.5:
            # 填充静音到0.5秒
            segment = torch.zeros(1, int(0.5 * audio_sr))
        else:
            segment = get_audio_segment(full_audio, audio_sr, start, end)
        # 转为单声道
        if segment.shape[0] > 1:
            segment = segment.mean(dim=0, keepdim=True)
        segment = segment.squeeze()
        if segment.dim() == 0:
            segment = segment.unsqueeze(0)
        audio_np = segment.cpu().numpy().astype(np.float32)
        # 重采样到16kHz
        if audio_sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=audio_sr, new_freq=16000)
            audio_np = resampler(torch.from_numpy(audio_np)).numpy()
        # 确保长度30秒
        audio_tensor = torch.from_numpy(audio_np)
        if audio_tensor.numel() < 480000:
            pad = torch.zeros(480000 - audio_tensor.numel())
            audio_tensor = torch.cat([audio_tensor, pad])
        else:
            audio_tensor = audio_tensor[:480000]
        mel = whisper.log_mel_spectrogram(audio_tensor)
        mel_batch = mel.unsqueeze(0).to(device)
        with torch.no_grad():
            features = whisper_model.encoder(mel_batch)
        emb = features.mean(dim=1).squeeze().cpu().numpy()
        audio_embs.append(emb)
    audio_embs = np.array(audio_embs)
    print(f"音频特征 shape: {audio_embs.shape}, 均值: {audio_embs.mean():.6f}, 标准差: {audio_embs.std():.6f}")

    # ---------- 语义合并 ----------
    print("6. 合并语义场景...")
    groups = merge_scenes(scenes, visual_embs, audio_embs, alpha=0.7, threshold=0.65)
    semantic = to_semantic(groups)

    SEMANTIC_JSON.write_text(
        json.dumps({"video": repaired_video.name, "semantic_scenes": semantic}, indent=2)
    )

    print(f"完成！原始场景数: {len(scenes)}, 语义场景数: {len(semantic)}")

if __name__ == "__main__":
    main()