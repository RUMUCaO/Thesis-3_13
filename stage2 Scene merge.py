import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import open_clip
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
model = model.to(device).eval()

def get_frame(video_path, t_sec):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_id = int(t_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

@torch.no_grad()
def encode_image(img):
    img = preprocess(img).unsqueeze(0).to(device)
    feat = model.encode_image(img)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy()

def load_scenes(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["scenes"]

def build_embeddings(video_path, scenes):
    embs = []

    for s in scenes:
        t = s["start_seconds"]
        img = get_frame(video_path, t)

        if img is None:
            continue

        emb = encode_image(img)
        embs.append(emb)

    return np.array(embs)

def merge_scenes(scenes, embeddings, sim_threshold=0.88):
    merged = []

    current_group = [scenes[0]]

    def cosine(a, b):
        return np.dot(a, b)

    for i in range(1, len(scenes)):
        sim = cosine(embeddings[i-1], embeddings[i])

        if sim > sim_threshold:
            current_group.append(scenes[i])
        else:
            merged.append(current_group)
            current_group = [scenes[i]]

    if current_group:
        merged.append(current_group)

    return merged

def to_semantic_scenes(groups):
    result = []

    for i, g in enumerate(groups):
        result.append({
            "index": i,
            "start": g[0]["start_seconds"],
            "end": g[-1]["end_seconds"],
            "scene_count": len(g)
        })

    return result

def main():
    video_path = Path("output_0.5x.mp4")
    scene_json_path = Path("stage2_TransNetV2_scenes.json")

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if not scene_json_path.exists():
        raise FileNotFoundError(scene_json_path)

    scenes = load_scenes(scene_json_path)

    embeddings = build_embeddings(video_path, scenes)

    groups = merge_scenes(
        scenes,
        embeddings,
        sim_threshold=0.88
    )

    semantic = to_semantic_scenes(groups)

    print("raw scenes:", len(scenes))
    print("semantic scenes:", len(semantic))

    Path("semantic_scenes.json").write_text(
        json.dumps(semantic, indent=2),
        encoding="utf-8"
    )
    
if __name__ == "__main__":
    main()