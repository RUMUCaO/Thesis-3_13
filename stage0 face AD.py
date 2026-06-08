#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Industrial Stable Face Identity Pipeline
- InsightFace (face embedding)
- IOU + Cosine tracking (online association)
- Track lifecycle management
- Optional DBSCAN re-id merge
"""

import cv2
import numpy as np
import torch
from collections import defaultdict
from sklearn.cluster import DBSCAN
import insightface


# =========================
# Utils
# =========================
def cosine_sim(a, b):
    return float(np.dot(a, b))


def iou(b1, b2):
    x1, y1, x2, y2 = b1
    x3, y3, x4, y4 = b2

    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (x2 - x1) * (y2 - y1)
    a2 = (x4 - x3) * (y4 - y3)

    return inter / (a1 + a2 - inter + 1e-6)


# =========================
# Main Stable Face Tracker
# =========================
def stable_face_id(video_path):

    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    tracks = []
    next_id = 0
    frame_id = 0

    # ===== hyperparameters =====
    IOU_TH = 0.3
    COS_TH = 0.5
    MAX_LOST_TIME = 2.5   # seconds

    def normalize(x):
        return x / (np.linalg.norm(x) + 1e-6)

    # =========================
    # STEP 1: TRACKING
    # =========================
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_id += 1
        t = frame_id / fps

        # 每秒采样（稳定 + 省算力）
        if frame_id % int(fps) != 0:
            continue

        faces = model.get(frame)

        detections = []
        for f in faces:
            bbox = f.bbox.astype(int).tolist()
            emb = normalize(f.embedding.astype(np.float32))
            detections.append((bbox, emb, t))

        # =========================
        # MATCHING
        # =========================
        for bbox, emb, t in detections:

            best_track = None
            best_score = -1

            for tr in tracks:
                if tr["lost"] and (t - tr["last_seen"]) > MAX_LOST_TIME:
                    continue

                sim = cosine_sim(emb, tr["emb"])
                overlap = iou(bbox, tr["bbox"])

                score = 0.6 * sim + 0.4 * overlap

                if sim > COS_TH and overlap > IOU_TH:
                    if score > best_score:
                        best_score = score
                        best_track = tr

            if best_track is not None:
                # ===== update track =====
                tr = best_track
                tr["emb"] = normalize(0.9 * tr["emb"] + 0.1 * emb)
                tr["bbox"] = bbox
                tr["times"].append(t)
                tr["last_seen"] = t
                tr["lost"] = False
                tr["frames"] += 1

            else:
                # ===== new track =====
                tracks.append({
                    "id": next_id,
                    "emb": emb,
                    "bbox": bbox,
                    "times": [t],
                    "frames": 1,
                    "last_seen": t,
                    "lost": False
                })
                next_id += 1

        # =========================
        # MARK LOST TRACKS
        # =========================
        for tr in tracks:
            if (t - tr["last_seen"]) > 1.0:
                tr["lost"] = True

    cap.release()

    # =========================
    # STEP 2: FILTER NOISE
    # =========================
    tracks = [t for t in tracks if len(t["times"]) >= 3]

    if not tracks:
        return []

    # =========================
    # STEP 3: OPTIONAL GLOBAL MERGE (re-id fix)
    # =========================
    X = np.array([t["emb"] for t in tracks])

    clustering = DBSCAN(
        eps=0.4,
        min_samples=2,
        metric="cosine"
    ).fit(X)

    clusters = defaultdict(list)

    for label, tr in zip(clustering.labels_, tracks):
        clusters[label].extend(tr["times"])

    # =========================
    # STEP 4: OUTPUT
    # =========================
    results = []
    for cid, ts in clusters.items():
        results.append({
            "cluster": int(cid),
            "start": float(min(ts)),
            "end": float(max(ts)),
            "duration": float(max(ts) - min(ts))
        })

    return results

def visualize_faces(video_path, output_path="face_vis.mp4"):
    import cv2
    import numpy as np
    import insightface
    from collections import defaultdict

    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    tracks = []
    next_id = 0

    def iou(a, b):
        x1,y1,x2,y2 = a
        x3,y3,x4,y4 = b
        xi1, yi1 = max(x1,x3), max(y1,y3)
        xi2, yi2 = min(x2,x4), min(y2,y4)
        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
        area1 = (x2-x1)*(y2-y1)
        area2 = (x4-x3)*(y4-y3)
        return inter / (area1 + area2 - inter + 1e-6)

    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_id += 1
        t = frame_id / fps

        faces = model.get(frame)

        detections = []

        for f in faces:
            bbox = f.bbox.astype(int).tolist()
            emb = f.embedding / (np.linalg.norm(f.embedding) + 1e-6)
            detections.append((bbox, emb))

        # match tracks
        for bbox, emb in detections:
            matched = False

            for tr in tracks:
                if iou(bbox, tr["bbox"]) > 0.3:
                    tr["emb"] = 0.9 * tr["emb"] + 0.1 * emb
                    tr["emb"] /= np.linalg.norm(tr["emb"]) + 1e-6
                    tr["bbox"] = bbox
                    tr["last_seen"] = t
                    tr["hits"] += 1
                    matched = True
                    tr["id"] = tr["id"]
                    break

            if not matched:
                tracks.append({
                    "id": next_id,
                    "bbox": bbox,
                    "emb": emb,
                    "last_seen": t,
                    "hits": 1
                })
                next_id += 1

        # draw
        for tr in tracks:
            if t - tr["last_seen"] > 2.0:
                continue

            x1,y1,x2,y2 = tr["bbox"]

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            cv2.putText(
                frame,
                f"Face_{tr['id']}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

        writer.write(frame)

    cap.release()
    writer.release()

    print("saved:", output_path)
    
# =========================
# TEST
# =========================
if __name__ == "__main__":
    video = r"Other\500D_clip_019.mp4"
    faces = stable_face_id(video)
    visualize_faces(video, "face_vis.mp4")

    print("faces:")
    print(len(faces))
    for f in faces:
        print(f)