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
    Use FFmpeg to force a re-encode of the video, repairing a corrupted H.264 data stream.

    Parameters:
    input_path: Path to the original video file

    output_path: Path to save the repaired video (if None, a temporary file will be automatically generated)

    crf: Quality parameter (default 23, lower values ​​result in better quality but larger file size)

    Return:
    Path to the repaired video file
    """
    if output_path is None:
        # Use temporary files to avoid contaminating the original files.
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="repaired_")
        output_path = Path(output_path)
        # Close the file descriptor; ffmpeg will overwrite it.
        import os
        os.close(fd)
    
    # Build the ffmpeg command: re-encode the video (H.264), audio (AAC), and overwrite the output
    cmd = [
        "ffmpeg", "-i", str(input_path),
        "-c:v", "libx264", "-crf", str(crf), "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        "-y", str(output_path)
    ]
    print(f"Video being repaired (re-encoded): {input_path.name} -> {output_path.name}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Repair complete")
        return output_path 
    except subprocess.CalledProcessError as e:
        print(f"Repair failed: {e}")
        raise

# =========================
# paths
# =========================
VIDEO_PATH = Path("THEGRAD.mp4")
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

def get_scenes_with_scenedetect(video_path: Path, threshold=15):
    """
    Use scenedetect to detect camera boundaries and return a list of scenes in a format compatible with TransNetV2.

    threshold: Content detection sensitivity (lower values ​​indicate greater sensitivity).
    """
    from scenedetect import detect, ContentDetector
    scene_list = detect(str(video_path), ContentDetector(threshold=threshold))
    scenes = []
    for i, (start_frame, end_frame) in enumerate(scene_list):
        scenes.append({
            "index": i,
            "start_frame": start_frame.frame_num,
            "end_frame": end_frame.frame_num,
            "start": start_frame.get_seconds(),
            "end": end_frame.get_seconds(),
        })
    return scenes

def extract_full_audio(video_path: Path, out_wav: str = "full_audio.wav"):
    """Extract the entire video's audio to a wav file, returning the audio array and sampling rate."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", "16000", out_wav
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    audio, sr = torchaudio.load(out_wav)
    Path(out_wav).unlink()  # Delete the temporary file
    return audio, sr

def get_audio_segment(full_audio, sr, start_sec, end_sec):
    """Extract a segment from the full audio."""
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    return full_audio[:, start_sample:end_sample]

# =========================
# multi-frame CLIP 
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
    ⭐ Key Upgrade: Scene-level embedding
    """
    start = scene["start"]
    end = scene["end"]

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
# semantic merge
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
        print(f"Pair {i-1}-{i} score: {score:.3f}")
        if score > threshold:
            cur.append(scenes[i])
        else:
            groups.append(cur)
            cur = [scenes[i]]
    groups.append(cur)
    return groups

def merge_scenes_clustering(
    scenes: List[Dict[str, Any]],
    visual_embs: np.ndarray,     # shape (N, D)
    audio_embs: np.ndarray,      # shape (N, D)
    alpha: float = 0.7,
    threshold: float = 0.85      # cosine distance threshold for stopping
) -> List[List[Dict[str, Any]]]:
    """
    Agglomerative clustering with average linkage, restricted to merging only
    temporally adjacent clusters. Uses a combined embedding (alpha * visual + (1-alpha) * audio)
    and stops when the minimum average cosine distance between any adjacent pair exceeds `threshold`.

    Parameters
    ----------
    scenes : list of dicts with at least "start", "end" keys.
    visual_embs : (N, D) array, one visual embedding per scene.
    audio_embs  : (N, D) array, one audio embedding per scene.
    alpha       : weight for visual modality.
    threshold   : cosine distance threshold (0..2). Default 0.85 corresponds to
                  a cosine similarity of ~0.15.

    Returns
    -------
    groups : list of list of scene dicts, each sublist a contiguous narrative segment.
    """
    n = len(scenes)
    if n == 0:
        return []

    # ----- 1. Build combined modality embedding -----
    # Normalise each modality embedding first (in case they aren't already)
    def l2_normalise(arr):
        norm = np.linalg.norm(arr, axis=1, keepdims=True)
        norm[norm == 0] = 1e-12
        return arr / norm

    v_norm = l2_normalise(np.asarray(visual_embs, dtype=float))
    a_norm = l2_normalise(np.asarray(audio_embs, dtype=float))
    combined = alpha * v_norm + (1 - alpha) * a_norm
    combined = l2_normalise(combined)          # keep on unit sphere

    # ----- 2. Cosine distance matrix -----
    # cosine_dist = 1 - dot(i,j)
    sim = combined @ combined.T                # (N, N) similarity matrix
    dist = 1.0 - sim                           # cosine distance matrix
    np.fill_diagonal(dist, np.inf)             # ignore self-distance

    # ----- 3. Cluster initialisation -----
    # Each scene is a cluster; store its start index (inclusive) and end index (exclusive).
    clusters = [{'start': i, 'end': i + 1} for i in range(n)]

    # ----- 4. Function to compute average cosine distance between two clusters -----
    def avg_dist(c1, c2):
        """Average cosine distance between all elements of c1 and c2."""
        slice_i = slice(c1['start'], c1['end'])
        slice_j = slice(c2['start'], c2['end'])
        # subset of the precomputed distance matrix
        sub = dist[slice_i, slice_j]
        return float(np.mean(sub))

    # ----- 5. Iterative merging -----
    while True:
        # find the pair of adjacent clusters with smallest average distance
        best_pair = None
        best_dist = np.inf
        for i in range(len(clusters) - 1):
            d = avg_dist(clusters[i], clusters[i + 1])
            if d < best_dist:
                best_dist = d
                best_pair = i

        # stop if the best distance exceeds the threshold
        if best_dist > threshold or best_pair is None:
            break

        # merge clusters[best_pair] and clusters[best_pair + 1]
        left = clusters[best_pair]
        right = clusters.pop(best_pair + 1)
        left['end'] = right['end']            # extend the left cluster

    # ----- 6. Convert cluster indices back to scene groups -----
    groups = []
    for cl in clusters:
        group_scenes = scenes[cl['start']:cl['end']]
        groups.append(group_scenes)

    return groups

def to_semantic(groups):
    return [
        {
            "index": i,
            "start": g[0]["start"],
            "end": g[-1]["end"],
            "scene_count": len(g)
        }
        for i, g in enumerate(groups)
    ]

# =========================
# main
# ========================= 
def main():
    print("Detecting scenes with scenedetect...")
    repaired_video_path = repair_video_with_ffmpeg(VIDEO_PATH)# Attempt to repair the video to avoid scene detection failures caused by decoding errors.
    scenes = get_scenes_with_scenedetect(repaired_video_path, threshold=15)   # Adjustable
    if not scenes:
        print("No scenes detected, abort.")
        return
    
    # Optional: Save the original scene (for easy review).
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
        start = s["start"]
        end = s["end"]
        if end - start < 0.5:   # For segments that are too short, directly assign the zero vector.
            audio_embs.append(np.zeros(512))
            continue
        
        segment = get_audio_segment(full_audio, audio_sr, start, end)
        
        # 1. Convert to mono (if the segment is multi-channel)
        if segment.dim() > 1 and segment.shape[0] > 1:
            segment = segment.mean(dim=0, keepdim=True).squeeze()
        else:
            segment = segment.squeeze()
        
        # 2. Convert to float32 numpy
        audio_np = segment.cpu().numpy().astype(np.float32)
        
        # 3. Resample to 16000 Hz (if the original sampling rate is not 16000).
        if audio_sr != 16000:
            import torchaudio
            # First convert back to tensor
            audio_tensor = torch.from_numpy(audio_np)
            # torchaudio resample
            resampler = torchaudio.transforms.Resample(orig_freq=audio_sr, new_freq=16000)
            audio_np = resampler(audio_tensor).numpy()
        
        # 4. Key: Pad or trim to 30 seconds (480000 samples)
        audio_30s = whisper.pad_or_trim(torch.from_numpy(audio_np))   # Output shape (480000,)
        
        # 5. Calculate the Mel spectrum (shape: 80, 3000)
        mel = whisper.log_mel_spectrogram(audio_30s)   # Note: Do not use unsqueeze here.
        
        # 6. Use the Whisper encoder to get the embedding (shape: 1500, 512)
        mel_batch = mel.unsqueeze(0).to(device)        # shape (1, 80, 3000)
        with torch.no_grad():
            features = whisper_model.encoder(mel_batch)   # shape (1, 1500, 512)
        
        # 7. Take the average of the time dimension as the fragment embedding
        emb = features.mean(dim=1).squeeze().cpu().numpy()  # shape (512,)
        emb = emb / np.linalg.norm(emb)
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