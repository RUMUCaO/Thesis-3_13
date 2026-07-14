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
    return result["segments"]   # Format compatible with WhisperX

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

    output = pipeline(audio_input)          # DiarizeOutput object
    diar = output.speaker_diarization       # Annotation object

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

    # Get audio duration
    waveform = audio_input["waveform"]
    sample_rate = audio_input["sample_rate"]
    audio_duration = waveform.shape[1] / sample_rate

    # Loading Model
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
    model.to(device)
    inference = Inference(model, window="whole")

    spk_embeddings = defaultdict(list)
    for turn, _, speaker in diar.itertracks(yield_label=True):
        if turn.duration < 0.5:
            continue
        # Boundary clipping to prevent overflow
        start = max(turn.start, 0.0)
        end = min(turn.end, audio_duration - 1e-6)
        if end <= start:
            continue
        safe_seg = Segment(start, end)
        try:
            emb = inference.crop(audio_input, safe_seg)
            spk_embeddings[speaker].append(emb)
        except Exception as e:
            print(f"Warning: Clip {start:.2f}-{end:.2f} Extraction failed: {e}")
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
    Face tracking is performed by sampling at the target frame rate (3 frames per second by default).
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
        fps = 25.0  # downgrade

    # Calculate the processing interval (e.g., fps=30, target_fps=3 → interval=10 frames).
    frame_interval = max(1, int(round(fps / target_fps)))

    tracks = []
    next_id = 0
    frame_id = 0   # Original frame count (for timestamp and loss detection)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_id / fps   # Current frame's real time (seconds)

        # Whether to process this frame (skipping frames)
        if frame_id % frame_interval == 0:
            faces = app.get(frame)

            for f in faces:
                emb = f.embedding
                bbox = f.bbox

                best_track = None
                best_score = 0
                for t in tracks:
                    # Tracks lost for more than 1 second will not be included in the matching process (tracks will be retained but not updated).
                    if frame_id - t["last_seen"] > fps * 3.0:
                        continue
                    sim = 1 - cosine(emb, t["embedding"])
                    iou_val = compute_iou(bbox, t["last_bbox"])
                    combined = 0.7 * sim + 0.3 * iou_val
                    if combined > best_score:
                        best_score = combined
                        best_track = t

                if best_score > 0.4:
                    # Match successful, update existing trajectory
                    t = best_track
                    t["embedding"] = 0.85 * t["embedding"] + 0.1 * emb
                    t["end_time"] = timestamp
                    t["embeddings"].append(emb)
                    t["last_seen"] = frame_id
                    t["last_bbox"] = bbox
                else:
                    # Create a new trajectory
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

    # Output result (consistent with the original logic)
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
    print("Check before returning:", type(face_tracks), len(face_tracks))
    if face_tracks:
        print("First element type:", type(face_tracks[0]))
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
        # Explicit loading model
        model = Model.from_pretrained(model_name, use_auth_token=hf_token)
        model.to(self.device)
        self.inference = Inference(model, window="whole")
        self._audio_input = None

    def set_audio(self, audio_input):
        self._audio_input = audio_input

    def extract(self, audio_input, start, end):
        if self._audio_input is None:
            raise ValueError("You must first call set_audio() to set the audio.")
        from pyannote.core import Segment
        waveform = self._audio_input["waveform"]
        sample_rate = self._audio_input["sample_rate"]
        duration = waveform.shape[1] / sample_rate
        # Boundary protection
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
    Returns joint_sim[i,j] ∈ [0,1], where a higher value indicates that spk_i and face_j are more likely to be the same person.
    """
    speakers = list({d["speaker"] for d in diar})
    face_ids = list({f["cluster"] for f in faces})
    n_spk, n_face = len(speakers), len(face_ids)
    S = np.zeros((n_spk, n_face))

    # Pre-calculate the total duration for each speaker (for overlap normalization).
    spk_duration = {}
    for d in diar:
        spk = d["speaker"]
        spk_duration[spk] = spk_duration.get(spk, 0.0) + (d["end"] - d["start"])
    
    # Pre-calculate the total duration for each face
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
        spk_emb = spk_mem.get(spk)          # It could be None
        spk_conf = spk_mem.confidence(spk) 

        for j, fid in enumerate(face_ids):
            face_emb = face_mem.get(fid)
            face_stab = next((f["stability"] for f in faces if f["cluster"] == fid), 0.0)
            
            # 1. Embedding similarity (probabilistic)
            if spk_emb is not None and face_emb is not None:
                cos_sim = 1 - cosine(spk_emb, face_emb)
                emb_prob = (cos_sim + 1) / 2   # Mapped to [0,1]
            else:
                emb_prob = 0.5

            # 2. Time overlap ratio (strictly normalized)
            total_overlap = 0.0
            for d in diar:
                if d["speaker"] != spk:
                    continue
                for f in faces:
                    if f["cluster"] != fid:
                        continue
                    # Overlapping time periods
                    start = max(d["start"], f.get("start_time", 0))
                    end = min(d["end"], f.get("end_time", float('inf')))
                    overlap = max(0, end - start)
                    total_overlap += overlap
            # Normalize using the maximum of the speaker's total duration and the face's total duration.
            denom = max(spk_duration.get(spk, 0.0), face_duration.get(fid, 0.0), 1e-6)
            overlap_prob = min(1.0, total_overlap / denom)

            # 3. Confidence product (probabilistic)
            conf_product = spk_conf * face_stab   # Both are near [0,1].
            conf_prob = 1.0 / (1.0 + np.exp(-5*(conf_product - 0.5)))  # sigmoid centralization

            # 4. Combinations (weighted geometric mean is more probabilistic, but linear is also okay)
            S[i, j] = 0.4 * emb_prob + 0.3 * overlap_prob + 0.3 * conf_prob

    # 5. Multiply by the co-occurrence prior (which has already been normalized).
    S = S * (cooccurrence + 1e-6)
    return S, speakers, face_ids

def sinkhorn_normalize(K, iters=10, eps=1e-6):
    "Double randomization normalization generates a joint probability distribution."
    K = K.copy()
    for _ in range(iters):
        K = K / (K.sum(axis=1, keepdims=True) + eps)
        K = K / (K.sum(axis=0, keepdims=True) + eps)
    return K

def global_assignment(diar, faces, spk_mem, face_mem, cooccurrence):
    S, speakers, face_ids = build_joint_similarity(diar, faces, spk_mem, face_mem, cooccurrence)
    P = sinkhorn_normalize(S)           # P[i,j] is the joint probability
    # Hard assignment: argmax (soft assignment can also be kept for EM)
    row_ind, col_ind = linear_sum_assignment(-P)   # Minimize negative probability
    mapping = {speakers[i]: face_ids[j] for i, j in zip(row_ind, col_ind)}
    return mapping, P

def probabilistic_merge(asr_segments, diar, mapping, P=None, temperature=1.0):
    """
    Associating ASR segments with speakers based on overlap ratio + softmax (with temperature):
    :param asr_segments: List of ASR segments, each containing start, end, and text
    :param diar: List of speaker time segments, each containing start, end, and speaker
    :param mapping: Hard mapping from speaker to face
    :param P: Joint probability matrix (unused, reserved interface)
    :param temperature: Softmax temperature; higher values ​​result in smoother text, lower values ​​result in sharper text.
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
                # Use the overlap ratio instead of absolute seconds.
                ratio = overlap / seg_dur
                spk_scores[d["speaker"]] += ratio
        
        if not spk_scores:
            best_spk = None
        else:
            scores = np.array(list(spk_scores.values()))
            #Application temperature: Divide by temperature and then softmax.
            if temperature != 1.0:
                scores = scores / temperature
            # Numerical stability softmax
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

    # Construct time interval
    spk_intervals = defaultdict(list)
    for d in diar:
        spk_intervals[d["speaker"]].append((d["start"], d["end"]))

    face_intervals = defaultdict(list)
    for f in faces:
        fid = f["cluster"]
        face_intervals[fid].append((f.get("start_time", 0), f.get("end_time", 0)))

    n_spk, n_face = len(speakers), len(face_ids)
    Q = np.zeros((n_spk, n_face))          # Initialize the posterior matrix

    # ---------- E-step: Computational posterior P(spk|face) ----------
    temperature = 0.5
    raw_scores = np.zeros((n_spk, n_face))
    for i, spk in enumerate(speakers):
        for j, fid in enumerate(face_ids):
            # 1. Time overlap
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

            # 2. Embedding similarity -> Likelihood
            spk_emb = spk_mem.get(spk)
            face_emb = face_mem.get(fid)
            if spk_emb is not None and face_emb is not None:
                cos_sim = 1 - cosine(spk_emb, face_emb)
                emb_lik = np.exp(cos_sim / temperature)
            else:
                emb_lik = 1.0
            time_factor = time_prob + 0.1
            raw_scores[i, j] = emb_lik * time_factor
            # 3. Combinatorial likelihood (time factor smoothing)
    prior = 1.0 / n_spk   # Uniform distribution
    posterior = raw_scores * prior   # Shape (n_spk, n_face)

    # Perform softmax on each face column (assuming uniform prior).
    for j in range(n_face):
        col = raw_scores[:, j]
        col = np.exp(col - np.max(col))
        posterior[:, j] = col / (col.sum() + 1e-8)
    # Perform softmax on each face (column) to obtain P(spk|face)
    Q = posterior   # Posterior matrix

    # ---------- M-step: Update Embedded ----------
    # Update speaker embedding (from audio clip)
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

    # Update the face embedding (from speaker memory).
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
    """Based on temporal overlap statistics, the co-occurrence intensity of each speaker and each face is analyzed; this is an optimized version."""
    # Group time ranges by speaker
    spk_intervals = defaultdict(list)
    for d in diar:
        spk_intervals[d["speaker"]].append((d["start"], d["end"]))
    
    # Group time ranges by face cluster
    face_intervals = defaultdict(list)
    for f in faces:
        fid = f["cluster"]
        start = f.get("start_time", 0)
        end = f.get("end_time", 0)
        if end > start:  # Only add valid intervals
            face_intervals[fid].append((start, end))
    
    speakers = list(spk_intervals.keys())
    face_ids = list(face_intervals.keys())
    C = np.zeros((len(speakers), len(face_ids)))
    
    # Helper function: calculate the total overlapping time of two interval lists
    def total_overlap(intervals_a, intervals_b):
        # Assume both lists are not sorted, sort them first
        a = sorted(intervals_a, key=lambda x: x[0])
        b = sorted(intervals_b, key=lambda x: x[0])
        i = j = 0
        total = 0.0
        while i < len(a) and j < len(b):
            # Calculate the overlap of the current two intervals
            start = max(a[i][0], b[j][0])
            end = min(a[i][1], b[j][1])
            if end > start:
                total += end - start
            # Move the pointer of the interval that ends earlier
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
    
    # Row normalization
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
            extract_audio(video, audio_path)          # Extract audio from video to a temporary file

            # ---- Preload audio (avoid torchcodec) ----
            waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)
            audio_input = {"waveform": waveform_tensor, "sample_rate": sr}

            # 1. Basic module (pass in audio_input instead of file path)
            diar_segments, diar_object = run_diarization(audio_input, hf_token, device)   # Speaker's time period
            asr = run_asr(audio_path, device)                      # ASR fragment (still using paths, because whisperx internally uses files)
            faces = face_tracking_with_time(video)                 # Face trajectory (including start/end)
            # Post-processing: Limiting the number of trajectories
            MAX_FACES = 300
            if len(faces) > MAX_FACES:
                print(f"Warning: Too many face trajectories detected ({len(faces)} > {MAX_FACES}), keeping the top 20 longest ones.")
                # Sort by duration in descending order and keep the top 20
                faces.sort(key=lambda x: x.get("duration", 0), reverse=True)
                faces = faces[:30]

            # 2. Memory Model
            spk_mem = SpeakerMemory()
            face_mem = FaceMemory()
            
            # Speaker encoder (requires modification to use the audio_input version, see below)
            speaker_encoder = SpeakerEncoder(device=device, hf_token=hf_token)
            speaker_encoder.set_audio(audio_input)   # New feature: Preset audio dictionary

            # Extract the speaker embedding (pass in audio_input instead of the path).
            speaker_embeddings = extract_speaker_embeddings_pyannote(audio_input, diar_object, device, hf_token)

            # ----- Initialize memory -----
            for face in faces:
                face_mem.update(face["cluster"], np.array(face["embedding"]))
            for spk_label, emb in speaker_embeddings.items():
                spk_id = int(spk_label.split("_")[-1])
                spk_mem.update(spk_id, emb)

            # 3. Constructing a co-occurrence matrix
            print("Type check:", type(faces), len(faces))
            if faces:
                print("First element type:", type(faces[0]))
                print("First element content:", faces[0])
            normalized_faces = []
            for f in faces:
                if isinstance(f, dict):
                    normalized_faces.append(f)
                elif isinstance(f, tuple):
                    # Assuming the tuple format is: (cluster, start_time, end_time, embedding, ...)

                    # Adjust the indexes based on the actual output (print it out first).）
                    print("Find the tuple format:", f)
                    # Temporary conversion (Example: Assume f[0]=cluster, f[1]=start_time, f[2]=end_time)
                    d = {
                        "cluster": f[0],
                        "start_time": f[1] if len(f) > 1 else 0,
                        "end_time": f[2] if len(f) > 2 else 0,
                        "embedding": f[3] if len(f) > 3 else None,
                    }
                    normalized_faces.append(d)
                else:
                    raise TypeError(f"Unknown type:{type(f)}")
            faces = normalized_faces
            cooccur, _, _ = build_cooccurrence_matrix(diar_segments, faces)

            # 5. Initial global matching
            mapping, P = global_assignment(diar_segments, faces, spk_mem, face_mem, cooccur)

            # 6. EM Iterative Optimization
            for _ in range(3):
                em_refine(
                    diar=diar_segments,
                    faces=faces,
                    mapping=mapping,
                    spk_mem=spk_mem,
                    face_mem=face_mem,
                    asr_segments=asr,
                    audio_input=audio_input,          # Pass audio_input instead
                    speaker_encoder=speaker_encoder
                )
                cooccur, _, _ = build_cooccurrence_matrix(diar_segments, faces)
                mapping, P = global_assignment(diar_segments, faces, spk_mem, face_mem, cooccur)

            # 7. Final merge
            merged = probabilistic_merge(asr, diar_segments, mapping, P, temperature=0.8)

            return {
                "diarization": diar_segments,
                "faces": faces,
                "mapping": mapping,
                "assignment_probs": P.tolist(),
                "result": merged
            }
    except Exception as e:
        print("Error occurred:", e)
        import traceback
        traceback.print_exc()
        raise   # It will still throw an error, but you can see the specific error.
# =========================
# MAIN
# ========================= 

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--video", default="THEGRAD.mp4")
    p.add_argument("--out", default="whisperx_results.json")
    p.add_argument('--hf-token', default=os.environ.get('HF_TOKEN', None))

    args = p.parse_args()

    out = run(args.video, args.hf_token)
    

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("saved:", args.out)