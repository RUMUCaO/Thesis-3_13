import cv2
import numpy as np
import insightface
from scipy.spatial.distance import cosine

# ----------------------------
# Initialize RetinaFace + ArcFace
# ----------------------------
arc_model = insightface.app.FaceAnalysis()
arc_model.prepare(ctx_id=0, det_size=(640, 640))

# ----------------------------
# Open video
# ----------------------------
video_path = "500D_clip_019.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Unable to open video file")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

# ----------------------------
# Parameters
# ----------------------------
FEATURE_THRESHOLD = 0.5 # Embedding matching threshold
CENTER_THRESHOLD = 100 # Optical flow center matching distance
MAX_MISSING_FRAMES = 10
DETECTION_INTERVAL = 5 # Perform face detection every few frames

tracked_faces = []  # each face: {id, embedding, bbox, center, gender, missing_frames, appear_times, points}
next_id = 1

# ----------------------------
# Functions
# ----------------------------
def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2)//2, (y1 + y2)//2)

def match_face(embedding, center, gender, frame_time):
    """Attempt to match existing tracked_faces; if no match is found, create a new ID."""
    global tracked_faces, next_id
    for face in tracked_faces:
        dist_feat = cosine(embedding, face['embedding'])
        dist_center = np.linalg.norm(np.array(center) - np.array(face['center']))
        if dist_feat < FEATURE_THRESHOLD or dist_center < CENTER_THRESHOLD:
            # Matched with the same person
            face['embedding'] = embedding  # Keyframe update embedding
            face['center'] = center
            w = face['bbox'][2] - face['bbox'][0]
            h = face['bbox'][3] - face['bbox'][1]
            face['bbox'] = [center[0]-w//2, center[1]-h//2, center[0]+w//2, center[1]+h//2]
            face['gender'] = gender
            face['missing_frames'] = 0
            face['appear_times'].append(frame_time)
            return face['id']
    # new ID
    new_id = next_id
    next_id += 1
    w, h = 100, 100  # initial size
    tracked_faces.append({
        'id': new_id,
        'embedding': embedding,
        'bbox': [center[0]-w//2, center[1]-h//2, center[0]+w//2, center[1]+h//2],
        'center': center,
        'gender': gender,
        'missing_frames': 0,
        'appear_times': [frame_time],
        'points': None
    })
    return new_id

# ----------------------------
# Optical flow params
# ----------------------------
lk_params = dict(winSize=(15,15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

prev_gray = None

# ----------------------------
# Main loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    frame_time = frame_count / fps
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ----------------------------
    # Keyframe detection
    # ----------------------------
    if frame_count % DETECTION_INTERVAL == 0 or prev_gray is None:
        faces = arc_model.get(frame)
        seen_ids = []

        for face in faces:
            x1, y1, x2, y2 = [int(b) for b in face.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            embedding = face.embedding
            gender = face.gender
            center = get_center([x1, y1, x2, y2])

            person_id = match_face(embedding, center, gender, frame_time)
            seen_ids.append(person_id)

        # Update missing_frames
        for face in tracked_faces:
            if face['id'] not in seen_ids:
                face['missing_frames'] += 1
        tracked_faces = [f for f in tracked_faces if f['missing_frames'] <= MAX_MISSING_FRAMES]

        # Update prev_gray
        prev_gray = gray.copy()

        # Key points for preparing optical flow
        for face in tracked_faces:
            x1, y1, x2, y2 = face['bbox']
            face_gray = prev_gray[y1:y2, x1:x2]
            if face_gray.size == 0:
                face['points'] = None
                continue
            pts = cv2.goodFeaturesToTrack(face_gray, mask=None, maxCorners=15,
                                          qualityLevel=0.01, minDistance=3)
            if pts is not None:
                pts += np.array([[x1, y1]], dtype=np.float32)  # coordinate offset
            face['points'] = pts

    # ----------------------------
    # Non-keyframe optical flow tracking
    # ----------------------------
    else:
        for face in tracked_faces:
            if face['points'] is None or len(face['points']) == 0:
                continue
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, face['points'].reshape(-1,1,2), None, **lk_params
            )
            if p1 is not None:
                good_points = p1[st.flatten()==1]
                if len(good_points) > 0:
                    mean_pt = np.mean(good_points, axis=0).ravel()
                    cx, cy = int(mean_pt[0]), int(mean_pt[1])
                    # Keep the bbox size
                    w = face['bbox'][2] - face['bbox'][0]
                    h = face['bbox'][3] - face['bbox'][1]
                    face['center'] = (cx, cy)
                    face['bbox'] = [cx - w//2, cy - h//2, cx + w//2, cy + h//2]
                    face['points'] = good_points.reshape(-1,2)
            face['missing_frames'] += 1
        prev_gray = gray.copy()

    # ----------------------------
    # Draw tracked faces
    # ----------------------------
    for face in tracked_faces:
        x1, y1, x2, y2 = face['bbox']
        gender_label = "Male" if face['gender']==1 else "Female"
        label = f"people{face['id']}, {gender_label}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Remove disappeared faces
    tracked_faces = [f for f in tracked_faces if f['missing_frames'] <= MAX_MISSING_FRAMES]

    cv2.imshow("Face Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ----------------------------
# Output timestamps
# ----------------------------
for face in tracked_faces:
    times = [round(t,2) for t in face['appear_times']]
    print(f"people{face['id']} ({'Male' if face['gender']==1 else 'Female'}) appears at seconds:", times)