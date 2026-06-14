#!/usr/bin/env python3
import os
import json
import tempfile
import subprocess
import numpy as np
import cv2
from polars import duration
import torch
import torchaudio
import soundfile as sf
from collections import defaultdict
from collections import deque
import scipy.signal
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
from pyannote.audio import Inference

# =========================
# AUDIO
# =========================

def extract_audio(video_path, out_audio_path):
    subprocess.check_call([
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1", "-ar", "16000", "-vn",
        out_audio_path
    ])

# =========================
# ASR (REAL WHISPERX)
# =========================

def run_asr(audio_path, device="cuda"):
    import whisper
    model = whisper.load_model("small", device=device)
    result = model.transcribe(audio_path, language="en")
    return result["segments"]   # 格式与 whisperx 兼容

# =========================
# DIARIZATION
# =========================

def run_diarization(audio_input, hf_token, device="cuda"):
    from pyannote.audio import Pipeline
    import torch

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1"
    )
    pipeline.to(torch.device(device))

    output = pipeline(audio_input)          # DiarizeOutput 对象
    diar = output.speaker_diarization       # Annotation 对象

    segments = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": int(speaker.split("_")[-1])
        })
    return segments, diar

def extract_speaker_embeddings_pyannote(audio_input, diar, device="cuda", hf_token=None):
    from pyannote.audio import Inference
    from pyannote.audio.core.model import Model
    import numpy as np
    from collections import defaultdict
    from pyannote.core import Segment

    # 获取音频时长
    waveform = audio_input["waveform"]
    sample_rate = audio_input["sample_rate"]
    audio_duration = waveform.shape[1] / sample_rate

    # 加载模型
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
    model.to(device)
    inference = Inference(model, window="whole")

    spk_embeddings = defaultdict(list)
    for turn, _, speaker in diar.itertracks(yield_label=True):
        if turn.duration < 0.5:
            continue
        # 边界裁剪，防止超出
        start = max(turn.start, 0.0)
        end = min(turn.end, audio_duration - 1e-6)
        if end <= start:
            continue
        safe_seg = Segment(start, end)
        try:
            emb = inference.crop(audio_input, safe_seg)
            spk_embeddings[speaker].append(emb)
        except Exception as e:
            print(f"警告：片段 {start:.2f}-{end:.2f} 提取失败: {e}")
            continue

    speaker_emb = {}
    for spk, emb_list in spk_embeddings.items():
        if emb_list:
            speaker_emb[spk] = np.mean(emb_list, axis=0)
    return speaker_emb

# =========================
# FACE REID + TRACKING
# =========================

def face_cluster(video_path):
    import insightface
    from scipy.spatial.distance import cosine

    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=0)

    cap = cv2.VideoCapture(video_path)

    tracks = []
    next_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = model.get(frame)

        for f in faces:
            emb = f.embedding

            best = None
            best_sim = 0

            for t in tracks:
                sim = 1 - cosine(emb, t["embedding"])

                if sim > best_sim:
                    best_sim = sim
                    best = t

            if best_sim > 0.6:
                best["embedding"] = 0.85 * best["embedding"] + 0.15 * emb
                best["count"] += 1
                best["last_seen"] = len(tracks)
            else:
                tracks.append({
                    "id": next_id,
                    "embedding": emb,
                    "count": 1,
                    "last_seen": len(tracks)
                })
                next_id += 1

    return [
        {
            "cluster": t["id"],
            "embedding": t["embedding"].tolist(),
            "stability": min(1.0, t["count"] / 30.0)
        }
        for t in tracks
    ]

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)
    
def face_tracking_with_time(video_path, target_fps=3):
    """
    人脸跟踪，按目标帧率采样处理（默认每秒3帧）
    """
    import insightface
    import cv2
    import numpy as np
    from scipy.spatial.distance import cosine
    from collections import deque

    app = insightface.app.FaceAnalysis()
    app.prepare(ctx_id=0)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # 降级

    # 计算处理间隔（例如 fps=30, target_fps=3 → 间隔=10帧）
    frame_interval = max(1, int(round(fps / target_fps)))

    tracks = []
    next_id = 0
    frame_id = 0   # 原始帧计数（用于时间戳和丢失判断）

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_id / fps   # 当前帧的真实时间（秒）

        # 是否处理这一帧（跳帧）
        if frame_id % frame_interval == 0:
            faces = app.get(frame)

            for f in faces:
                emb = f.embedding
                bbox = f.bbox

                best_track = None
                best_score = 0
                for t in tracks:
                    # 丢失超过1秒的轨迹暂不参与匹配（保留轨迹但不更新）
                    if frame_id - t["last_seen"] > fps * 3.0:
                        continue
                    sim = 1 - cosine(emb, t["embedding"])
                    iou_val = compute_iou(bbox, t["last_bbox"])
                    combined = 0.7 * sim + 0.3 * iou_val
                    if combined > best_score:
                        best_score = combined
                        best_track = t

                if best_score > 0.4:
                    # 匹配成功，更新已有轨迹
                    t = best_track
                    t["embedding"] = 0.85 * t["embedding"] + 0.1 * emb
                    t["end_time"] = timestamp
                    t["embeddings"].append(emb)
                    t["last_seen"] = frame_id
                    t["last_bbox"] = bbox
                else:
                    # 创建新轨迹
                    tracks.append({
                        "id": next_id,
                        "embedding": emb,
                        "embeddings": deque([emb], maxlen=30),
                        "start_time": timestamp,
                        "end_time": timestamp,
                        "last_seen": frame_id,
                        "last_bbox": bbox,
                    })
                    next_id += 1

        frame_id += 1

    cap.release()

    # 输出结果（与原逻辑一致）
    face_tracks = []
    for t in tracks:
        stability = min(1.0, len(t["embeddings"]) / 30.0)
        face_tracks.append({
            "cluster": t["id"],
            "id": t["id"],
            "start_time": t["start_time"],
            "end_time": t["end_time"],
            "duration": t["end_time"] - t["start_time"],
            "embedding": t["embedding"].tolist(),
            "stability": stability
        })
    print("返回前检查:", type(face_tracks), len(face_tracks))
    if face_tracks:
        print("首元素类型:", type(face_tracks[0]))
    return face_tracks

# =========================
# MEMORY MODELS
# =========================

class SpeakerMemory:
    def __init__(self, dim=256, queue_size=20):
        self.ema = {}
        self.var = {}
        self.conf = {}
        self.queue = {}
        self.queue_size = queue_size

    def update(self, spk, emb):

        emb = np.array(emb)

        if spk not in self.ema:
            self.ema[spk] = emb
            self.var[spk] = np.zeros_like(emb)
            self.conf[spk] = 0.3
            self.queue[spk] = deque(maxlen=self.queue_size)
            self.queue[spk].append(emb)
            return

        old = self.ema[spk]

        # EMA update
        new_ema = 0.9 * old + 0.1 * emb
        self.ema[spk] = new_ema

        # variance update (uncertainty tracking)
        diff = emb - old
        self.var[spk] = 0.9 * self.var[spk] + 0.1 * (diff ** 2)

        # confidence = inverse uncertainty
        self.conf[spk] = 1.0 / (1.0 + np.mean(self.var[spk]))

        # temporal queue
        self.queue[spk].append(emb)

    def get(self, spk):
        return self.ema.get(spk)

    def uncertainty(self, spk):
        return np.mean(self.var.get(spk, np.zeros(1)))

    def confidence(self, spk):
        return self.conf.get(spk, 0.0)

    def history(self, spk):
        return list(self.queue.get(spk, []))

class FaceMemory:
    def __init__(self, queue_size=30):
        self.ema = {}
        self.var = {}
        self.stability = {}
        self.queue = {}
        self.queue_size = queue_size
        self.count = {}

    def update(self, fid, emb):

        emb = np.array(emb)

        if fid not in self.ema:
            self.ema[fid] = emb
            self.var[fid] = np.zeros_like(emb)
            self.queue[fid] = deque(maxlen=self.queue_size)
            self.count[fid] = 1
            self.stability[fid] = 0.3
            self.queue[fid].append(emb)
            return

        old = self.ema[fid]

        # EMA
        self.ema[fid] = 0.9 * old + 0.1 * emb

        # variance
        diff = emb - old
        self.var[fid] = 0.9 * self.var[fid] + 0.1 * (diff ** 2)

        # stability = function of track length + variance
        self.count[fid] += 1

        self.stability[fid] = (
            1 - np.exp(-self.count[fid] / 20.0)
        ) * (1 / (1 + np.mean(self.var[fid])))

        self.queue[fid].append(emb)

    def get(self, fid):
        return self.ema.get(fid)

    def stability_score(self, fid):
        return self.stability.get(fid, 0.0)

class SpeakerTemporalModel:
    def __init__(self, alpha=0.8):
        self.alpha = alpha
        self.prev_spk = None
        self.transition = {}  # P(s_t | s_{t-1})

    def score(self, spk):
        if self.prev_spk is None:
            return 1.0

        key = (self.prev_spk, spk)
        return self.transition.get(key, 0.5)

    def update(self, spk):
        if self.prev_spk is not None:
            key = (self.prev_spk, spk)
            self.transition[key] = self.alpha * self.transition.get(key, 0.5) + (1 - self.alpha)

        self.prev_spk = spk
    
class FaceTemporalModel:
    def __init__(self):
        self.last_emb = {}
        self.motion_score = {}

    def update(self, fid, emb):

        emb = np.array(emb)

        if fid not in self.last_emb:
            self.last_emb[fid] = emb
            self.motion_score[fid] = 1.0
            return

        prev = self.last_emb[fid]

        # motion consistency = similarity between consecutive embeddings
        sim = 1 - cosine(prev, emb)

        self.motion_score[fid] = 0.8 * self.motion_score[fid] + 0.2 * sim

        self.last_emb[fid] = emb

    def score(self, fid):
        return self.motion_score.get(fid, 0.5)
    
class SpeakerEncoder:
    def __init__(self, model_name="pyannote/embedding", device="cuda", hf_token=None):
        from pyannote.audio import Inference
        from pyannote.audio.core.model import Model
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        # 显式加载模型
        model = Model.from_pretrained(model_name, use_auth_token=hf_token)
        model.to(self.device)
        self.inference = Inference(model, window="whole")
        self._audio_input = None

    def set_audio(self, audio_input):
        self._audio_input = audio_input

    def extract(self, audio_input, start, end):
        if self._audio_input is None:
            raise ValueError("必须先调用 set_audio() 设置音频")
        from pyannote.core import Segment
        waveform = self._audio_input["waveform"]
        sample_rate = self._audio_input["sample_rate"]
        duration = waveform.shape[1] / sample_rate
        # 边界保护
        start = max(0.0, min(start, duration - 1e-6))
        end = max(start, min(end, duration - 1e-6))
        from pyannote.core import Segment
        segment = Segment(start, end)
        emb = self.inference.crop(self._audio_input, segment)
        emb = np.array(emb)
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        return emb

def smooth_segments(segments, window=3):
    smoothed = []

    for i in range(len(segments)):
        start = max(0, i - window)
        end = min(len(segments), i + window + 1)

        chunk = segments[start:end]

        avg_start = np.mean([c["start"] for c in chunk])
        avg_end = np.mean([c["end"] for c in chunk])

        smoothed.append({
            "text": segments[i]["text"],
            "start": avg_start,
            "end": avg_end
        })

    return smoothed
# =========================
# ALIGNMENT SCORE (GLOBAL)
# =========================

def norm_cosine(cos_sim):
    # cos_sim ∈ [-1,1]
    return (cos_sim + 1) / 2

def norm_overlap(overlap, spk_dur, face_dur):
    denom = max(spk_dur, face_dur, 1e-6)
    return min(1.0, overlap / denom)

def norm_conf(conf):
    # conf already unstable
    return 1 / (1 + np.exp(-5 * (conf - 0.5)))

def norm_stability(x):
    return 1 - np.exp(-x / 20.0)

def unified_score(spk_emb, face_emb,
                  overlap,
                  spk_dur,
                  face_dur,
                  spk_conf,
                  face_conf):

    emb = norm_cosine(1 - cosine(spk_emb, face_emb))
    ov  = norm_overlap(overlap, spk_dur, face_dur)
    sc  = norm_conf(spk_conf)
    fc  = norm_conf(face_conf)

    # probabilistic fusion (NOT linear heuristic)
    score = emb * 0.5 + ov * 0.3 + (sc * fc) * 0.2

    return score

# =========================
# ASSIGNMENT (HUNGARIAN)
# =========================

def build_joint_similarity(diar, faces, spk_mem, face_mem, cooccurrence):
    """
    返回 joint_sim[i,j] ∈ [0,1]，越高表示 spk_i 与 face_j 越可能同一人
    """
    speakers = list({d["speaker"] for d in diar})
    face_ids = list({f["cluster"] for f in faces})
    n_spk, n_face = len(speakers), len(face_ids)
    S = np.zeros((n_spk, n_face))

    # 预计算每个 speaker 的总时长（用于重叠归一化）
    spk_duration = {}
    for d in diar:
        spk = d["speaker"]
        spk_duration[spk] = spk_duration.get(spk, 0.0) + (d["end"] - d["start"])
    
    # 预计算每个 face 的总时长
    face_duration = {}
    for f in faces:
        fid = f["cluster"]
        if "duration" in f:
            dur = f["duration"]
        else:
            start = f.get("start_time", 0)
            end = f.get("end_time", 0)
            dur = max(0, end - start)
        if dur > 0:
            face_duration[fid] = face_duration.get(fid, 0) + dur

    for i, spk in enumerate(speakers):
        spk_emb = spk_mem.get(spk)          # 可能为 None
        spk_conf = spk_mem.confidence(spk)  # 已修复命名

        for j, fid in enumerate(face_ids):
            face_emb = face_mem.get(fid)
            face_stab = next((f["stability"] for f in faces if f["cluster"] == fid), 0.0)
            
            # 1. 嵌入相似度（概率化）
            if spk_emb is not None and face_emb is not None:
                cos_sim = 1 - cosine(spk_emb, face_emb)
                emb_prob = (cos_sim + 1) / 2   # 映射到 [0,1]
            else:
                emb_prob = 0.5

            # 2. 时间重叠比例（严格归一化）
            total_overlap = 0.0
            for d in diar:
                if d["speaker"] != spk:
                    continue
                for f in faces:
                    if f["cluster"] != fid:
                        continue
                    # 时间段重叠
                    start = max(d["start"], f.get("start_time", 0))
                    end = min(d["end"], f.get("end_time", float('inf')))
                    overlap = max(0, end - start)
                    total_overlap += overlap
            # 用 speaker 总时长和 face 总时长的最大值归一化
            denom = max(spk_duration.get(spk, 0.0), face_duration.get(fid, 0.0), 1e-6)
            overlap_prob = min(1.0, total_overlap / denom)

            # 3. 置信度乘积（概率化）
            conf_product = spk_conf * face_stab   # 两者都在 [0,1] 附近
            conf_prob = 1.0 / (1.0 + np.exp(-5*(conf_product - 0.5)))  # sigmoid 中心化

            # 4. 组合（加权几何平均更概率化，但线性也 OK）
            S[i, j] = 0.4 * emb_prob + 0.3 * overlap_prob + 0.3 * conf_prob

    # 5. 乘上 co-occurrence 先验（已经归一化）
    S = S * (cooccurrence + 1e-6)
    return S, speakers, face_ids

def sinkhorn_normalize(K, iters=10, eps=1e-6):
    """双随机归一化，产生联合概率分布"""
    K = K.copy()
    for _ in range(iters):
        K = K / (K.sum(axis=1, keepdims=True) + eps)
        K = K / (K.sum(axis=0, keepdims=True) + eps)
    return K

def global_assignment(diar, faces, spk_mem, face_mem, cooccurrence):
    S, speakers, face_ids = build_joint_similarity(diar, faces, spk_mem, face_mem, cooccurrence)
    P = sinkhorn_normalize(S)           # P[i,j] 是联合概率
    # 硬分配：argmax（也可保留软分配用于 EM）
    row_ind, col_ind = linear_sum_assignment(-P)   # 最小化负概率
    mapping = {speakers[i]: face_ids[j] for i, j in zip(row_ind, col_ind)}
    return mapping, P

def probabilistic_merge(asr_segments, diar, mapping, P=None, temperature=1.0):
    """
    将 ASR 片段与说话人关联，基于重叠比例 + softmax (带温度)
    :param asr_segments: ASR 片段列表，每个包含 start, end, text
    :param diar: 说话人时段列表，每个包含 start, end, speaker
    :param mapping: speaker -> face 的硬映射
    :param P: 联合概率矩阵 (未使用，保留接口)
    :param temperature: softmax 温度，越大越平滑，越小越尖锐
    """
    spk_to_face = mapping
    out = []
    for seg in asr_segments:
        seg_start, seg_end = seg["start"], seg["end"]
        seg_dur = seg_end - seg_start
        if seg_dur <= 0:
            continue
        
        spk_scores = defaultdict(float)
        for d in diar:
            overlap = max(0, min(seg_end, d["end"]) - max(seg_start, d["start"]))
            if overlap > 0:
                # 使用重叠比例，而不是绝对秒数
                ratio = overlap / seg_dur
                spk_scores[d["speaker"]] += ratio
        
        if not spk_scores:
            best_spk = None
        else:
            scores = np.array(list(spk_scores.values()))
            # 应用温度：除以 temperature 后再 softmax
            if temperature != 1.0:
                scores = scores / temperature
            # 数值稳定 softmax
            max_score = np.max(scores)
            exp_scores = np.exp(scores - max_score)
            prob = exp_scores / (np.sum(exp_scores) + 1e-8)
            best_idx = np.argmax(prob)
            best_spk = list(spk_scores.keys())[best_idx]
        
        if best_spk is None:
            face = None
        else:
            face = spk_to_face.get(best_spk)
        
        out.append({
            "text": seg["text"],
            "start": seg_start,
            "end": seg_end,
            "speaker": best_spk,
            "character": face,
        })
    return out
# =========================
# EM REFINE (REAL LOOP)
# =========================

def em_refine(diar, faces, mapping, spk_mem, face_mem, asr_segments, audio_input, speaker_encoder):
    speakers = list({d["speaker"] for d in diar})
    face_ids = list(set(f["cluster"] for f in faces))

    # 构建时间区间
    spk_intervals = defaultdict(list)
    for d in diar:
        spk_intervals[d["speaker"]].append((d["start"], d["end"]))

    face_intervals = defaultdict(list)
    for f in faces:
        fid = f["cluster"]
        face_intervals[fid].append((f.get("start_time", 0), f.get("end_time", 0)))

    n_spk, n_face = len(speakers), len(face_ids)
    Q = np.zeros((n_spk, n_face))          # 初始化后验矩阵

    # ---------- E-step: 计算后验 P(spk|face) ----------
    temperature = 0.5
    raw_scores = np.zeros((n_spk, n_face))
    for i, spk in enumerate(speakers):
        for j, fid in enumerate(face_ids):
            # 1. 时间重叠度
            overlap_sum = 0.0
            for s_start, s_end in spk_intervals[spk]:
                for f_start, f_end in face_intervals[fid]:
                    start = max(s_start, f_start)
                    end = min(s_end, f_end)
                    overlap_sum += max(0, end - start)
            spk_total = sum(e - s for s, e in spk_intervals[spk])
            face_total = sum(e - s for s, e in face_intervals[fid])
            denom = max(spk_total, face_total, 1e-6)
            time_prob = min(1.0, overlap_sum / denom)

            # 2. 嵌入相似度 -> 似然
            spk_emb = spk_mem.get(spk)
            face_emb = face_mem.get(fid)
            if spk_emb is not None and face_emb is not None:
                cos_sim = 1 - cosine(spk_emb, face_emb)
                emb_lik = np.exp(cos_sim / temperature)
            else:
                emb_lik = 1.0
            time_factor = time_prob + 0.1
            raw_scores[i, j] = emb_lik * time_factor
            # 3. 组合似然（时间因子平滑）
    prior = 1.0 / n_spk   # 均匀分布
    posterior = raw_scores * prior   # 形状 (n_spk, n_face)

    # 对每个 face 列做 softmax（假设均匀先验）
    for j in range(n_face):
        col = raw_scores[:, j]
        col = np.exp(col - np.max(col))
        posterior[:, j] = col / (col.sum() + 1e-8)
    # 对每个 face (列) 做 softmax，得到 P(spk|face)\
    Q = posterior   # 后验矩阵

    # ---------- M-step: 更新嵌入 ----------
    # 更新 speaker 嵌入（从音频片段）
    for i, spk in enumerate(speakers):
        weighted_emb = None
        total_weight = 0.0
        for seg in asr_segments:
            seg_start, seg_end = seg["start"], seg["end"]
            for j, fid in enumerate(face_ids):
                weight = Q[i, j]
                if weight > 0.3:
                    audio_emb = speaker_encoder.extract(audio_input, seg_start, seg_end)
                    if weighted_emb is None:
                        weighted_emb = weight * audio_emb
                    else:
                        weighted_emb += weight * audio_emb
                    total_weight += weight
        if total_weight > 0 and weighted_emb is not None:
            new_emb = weighted_emb / total_weight
            spk_mem.update(spk, new_emb)

    # 更新 face 嵌入（从 speaker 记忆）
    for j, fid in enumerate(face_ids):
        weighted_emb = None
        total_weight = 0.0
        for i, spk in enumerate(speakers):
            weight = Q[i, j]
            spk_emb = spk_mem.get(spk)
            if spk_emb is not None and weight > 0:
                if weighted_emb is None:
                    weighted_emb = weight * spk_emb
                else:
                    weighted_emb += weight * spk_emb
                total_weight += weight
        if weighted_emb is not None and total_weight > 0:
            new_emb = weighted_emb / total_weight
            face_mem.update(fid, new_emb)
                
def build_cooccurrence_matrix(diar, faces):
    """基于时间重叠统计每个 speaker 与每个 face 的共现强度，优化版"""
    # 按 speaker 分组时间区间
    spk_intervals = defaultdict(list)
    for d in diar:
        spk_intervals[d["speaker"]].append((d["start"], d["end"]))
    
    # 按 face cluster 分组时间区间
    face_intervals = defaultdict(list)
    for f in faces:
        fid = f["cluster"]
        start = f.get("start_time", 0)
        end = f.get("end_time", 0)
        if end > start:  # 只添加有效区间
            face_intervals[fid].append((start, end))
    
    speakers = list(spk_intervals.keys())
    face_ids = list(face_intervals.keys())
    C = np.zeros((len(speakers), len(face_ids)))
    
    # 辅助函数：计算两个区间列表的总重叠时间
    def total_overlap(intervals_a, intervals_b):
        # 假设两个列表未排序，先排序
        a = sorted(intervals_a, key=lambda x: x[0])
        b = sorted(intervals_b, key=lambda x: x[0])
        i = j = 0
        total = 0.0
        while i < len(a) and j < len(b):
            # 计算当前两个区间的重叠
            start = max(a[i][0], b[j][0])
            end = min(a[i][1], b[j][1])
            if end > start:
                total += end - start
            # 移动结束时间较早的指针
            if a[i][1] < b[j][1]:
                i += 1
            else:
                j += 1
        return total
    
    for i, spk in enumerate(speakers):
        spk_ivals = spk_intervals[spk]
        for j, fid in enumerate(face_ids):
            face_ivals = face_intervals[fid]
            overlap = total_overlap(spk_ivals, face_ivals)
            C[i, j] = overlap
    
    # 行归一化
    row_sums = C.sum(axis=1, keepdims=True)
    C = C / (row_sums + 1e-6)
    return C, speakers, face_ids

# =========================
# MERGE
# =========================

def temporal_score(
    emb_sim,
    spk_continuity,
    face_motion,
    time_overlap
):

    return (
        0.4 * emb_sim +
        0.3 * spk_continuity +
        0.2 * face_motion +
        0.1 * time_overlap
    )

def merge(asr_segments, diar, mapping):
    def find_spk(seg):
        best, best_ov = None, 0
        for d in diar:
            ov = max(0, min(seg["end"], d["end"]) - max(seg["start"], d["start"]))
            if ov > best_ov:
                best = d["speaker"]
                best_ov = ov
        return best

    out = []

    for s in asr_segments:
        spk = find_spk(s)
        face = mapping.get(spk)

        out.append({
            "text": s["text"],
            "start": s["start"],
            "end": s["end"],
            "speaker": spk,
            "character": face
        })

    return out

# =========================
# PIPELINE (EM LOOP)
# =========================

def run(video, hf_token, device="cuda"):
    try:
        import librosa
        import torch
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as td:
            audio_path = os.path.join(td, "a.wav")
            extract_audio(video, audio_path)          # 从视频中提取音频到临时文件

            # ---- 预加载音频（避开 torchcodec）----
            waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)
            audio_input = {"waveform": waveform_tensor, "sample_rate": sr}

            # 1. 基础模块（传入 audio_input 而不是文件路径）
            diar_segments, diar_object = run_diarization(audio_input, hf_token, device)   # 说话人时段
            asr = run_asr(audio_path, device)                      # ASR 片段（仍用路径，因为 whisperx 内部用文件）
            faces = face_tracking_with_time(video)                 # 人脸轨迹（含 start/end）
            # 后处理：限制轨迹数量
            MAX_FACES = 300
            if len(faces) > MAX_FACES:
                print(f"警告：检测到过多的人脸轨迹 ({len(faces)} > {MAX_FACES})，将保留持续时间最长的前20个轨迹。")
                # 按持续时间降序排序，取前20
                faces.sort(key=lambda x: x.get("duration", 0), reverse=True)
                faces = faces[:30]

            # 2. 记忆模型
            spk_mem = SpeakerMemory()
            face_mem = FaceMemory()
            
            # 说话人编码器（需要修改为使用 audio_input 的版本，见下文）
            speaker_encoder = SpeakerEncoder(device=device, hf_token=hf_token)
            speaker_encoder.set_audio(audio_input)   # 新增方法：预设置音频字典

            # 提取说话人嵌入（传入 audio_input 而非路径）
            speaker_embeddings = extract_speaker_embeddings_pyannote(audio_input, diar_object, device, hf_token)

            # ----- 初始化记忆 -----
            for face in faces:
                face_mem.update(face["cluster"], np.array(face["embedding"]))
            for spk_label, emb in speaker_embeddings.items():
                spk_id = int(spk_label.split("_")[-1])
                spk_mem.update(spk_id, emb)

            # 3. 构建共现矩阵
            print("类型检查：", type(faces), len(faces))
            if faces:
                print("第一个元素类型：", type(faces[0]))
                print("第一个元素内容：", faces[0])
            normalized_faces = []
            for f in faces:
                if isinstance(f, dict):
                    normalized_faces.append(f)
                elif isinstance(f, tuple):
                    # 假设元组格式：(cluster, start_time, end_time, embedding, ...)
                    # 根据实际输出调整索引（先打印看看）
                    print("发现元组格式:", f)
                    # 临时转换（示例：假设 f[0]=cluster, f[1]=start_time, f[2]=end_time）
                    d = {
                        "cluster": f[0],
                        "start_time": f[1] if len(f) > 1 else 0,
                        "end_time": f[2] if len(f) > 2 else 0,
                        "embedding": f[3] if len(f) > 3 else None,
                    }
                    normalized_faces.append(d)
                else:
                    raise TypeError(f"未知类型: {type(f)}")
            faces = normalized_faces
            cooccur, _, _ = build_cooccurrence_matrix(diar_segments, faces)

            # 5. 初始全局匹配
            mapping, P = global_assignment(diar_segments, faces, spk_mem, face_mem, cooccur)

            # 6. EM 迭代优化
            for _ in range(3):
                em_refine(
                    diar=diar_segments,
                    faces=faces,
                    mapping=mapping,
                    spk_mem=spk_mem,
                    face_mem=face_mem,
                    asr_segments=asr,
                    audio_input=audio_input,          # 改为传递 audio_input
                    speaker_encoder=speaker_encoder
                )
                cooccur, _, _ = build_cooccurrence_matrix(diar_segments, faces)
                mapping, P = global_assignment(diar_segments, faces, spk_mem, face_mem, cooccur)

            # 7. 最终合并
            merged = probabilistic_merge(asr, diar_segments, mapping, P, temperature=0.8)

            return {
                "diarization": diar_segments,
                "faces": faces,
                "mapping": mapping,
                "assignment_probs": P.tolist(),
                "result": merged
            }
    except Exception as e:
        print("发生错误:", e)
        import traceback
        traceback.print_exc()
        raise   # 仍然抛出，但你能看到具体错误
# =========================
# MAIN
# ========================= 

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--video", default="PW.mp4")
    p.add_argument("--out", default="whisperx_results.json")
    p.add_argument('--hf-token', default=os.environ.get('HF_TOKEN', None))

    args = p.parse_args()

    out = run(args.video, args.hf_token)
    

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("saved:", args.out)