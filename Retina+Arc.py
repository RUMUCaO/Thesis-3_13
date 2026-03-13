import cv2
import numpy as np
import insightface
from scipy.spatial.distance import cosine

# ----------------------------
# Initialize RetinaFace + ArcFace
# ----------------------------
arc_model = insightface.app.FaceAnalysis()
arc_model.prepare(ctx_id=-1, det_size=(640, 640))

# ----------------------------
# Open video
# ----------------------------
video_path = "500D_clip_019.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Unable to open video file")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)  # Get FPS for timestamp calculation
frame_count = 0

# ----------------------------
# Parameter settings
# ----------------------------
FEATURE_THRESHOLD = 0.6
CENTER_THRESHOLD = 50
MAX_MISSING_FRAMES = 10

tracked_faces = []  # Each element: dict {id, embedding, center, gender, missing_frames, appear_times}
next_id = 1

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def match_face(embedding, center, gender, frame_time):
    global tracked_faces, next_id
    for face in tracked_faces:
        dist_feat = cosine(embedding, face['embedding'])
        dist_center = np.linalg.norm(np.array(center) - np.array(face['center']))
        if dist_feat < FEATURE_THRESHOLD and dist_center < CENTER_THRESHOLD:
            face['embedding'] = embedding
            face['center'] = center
            face['gender'] = gender
            face['missing_frames'] = 0
            face['appear_times'].append(frame_time)
            return face['id']
    # New ID
    new_id = next_id
    next_id += 1
    tracked_faces.append({
        'id': new_id,
        'embedding': embedding,
        'center': center,
        'gender': gender,
        'missing_frames': 0,
        'appear_times': [frame_time]
    })
    return new_id

# ----------------------------
# Process frame by frame
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_time = frame_count / fps  # Seconds

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

        gender_label = "Male" if gender == 1 else "Female"
        label = f"people{person_id}, {gender_label}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Update missing_frames
    for face in tracked_faces:
        if face['id'] not in seen_ids:
            face['missing_frames'] += 1

    # Remove IDs that have been missing for too long
    tracked_faces = [f for f in tracked_faces if f['missing_frames'] <= MAX_MISSING_FRAMES]

    cv2.imshow("RetinaFace + ArcFace Label + Time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ----------------------------
# Output appearance timestamps for each ID
# ----------------------------
for face in tracked_faces:
    appear_times_sec = [round(t, 2) for t in face['appear_times']]
    print(f"people{face['id']} ({'Male' if face['gender']==1 else 'Female'}) appears at seconds:", appear_times_sec)