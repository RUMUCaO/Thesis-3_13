import cv2
import numpy as np
import insightface
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment

# ----------------------------
# Initialize model
# ----------------------------
arc_model = insightface.app.FaceAnalysis()
arc_model.prepare(ctx_id=0, det_size=(640, 640))

# ----------------------------
# Video
# ----------------------------
video_path = "500D_clip_019.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

# ----------------------------
# Parameters
# ----------------------------
FEATURE_THRESHOLD = 0.5
REID_THRESHOLD = 0.4
MAX_MISSING = 40
EMB_HISTORY = 10
MAIN_CHAR_BONUS = 0.7   # Make the main character easier to match
STEAL_PENALTY = 0.3     # Prevent ID stealing
SPATIAL_TOLERANCE = 150 # Spatial tolerance for short-term continuity merge

next_id = 1
active_tracks = []
lost_tracks = []
identity_bank = []

main_id = None

# ----------------------------
# Utils
# ----------------------------
def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1+x2)//2, (y1+y2)//2])

def update_embedding(track, emb):
    track['emb_list'].append(emb)
    if len(track['emb_list']) > EMB_HISTORY:
        track['emb_list'].pop(0)
    track['embedding'] = np.mean(track['emb_list'], axis=0)

def create_track(emb, bbox, gender, time):
    global next_id
    t = {
        'id': next_id,
        'embedding': emb,
        'emb_list':[emb],
        'bbox':bbox,
        'center':get_center(bbox),
        'prev_center':get_center(bbox),
        'gender':gender,
        'missing':0,
        'times':[time]
    }
    next_id += 1
    return t

# ----------------------------
# Main loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    t = frame_count / fps

    faces = arc_model.get(frame)

    detections = []
    for f in faces:
        x1,y1,x2,y2 = map(int,f.bbox)
        emb = f.embedding
        gender = f.gender
        detections.append((np.array([x1,y1,x2,y2]), emb, gender))

    # ----------------------------
    # 🎯 Main character identification
    # ----------------------------
    if len(active_tracks) > 0:
        scores = []
        for tr in active_tracks:
            area = (tr['bbox'][2]-tr['bbox'][0]) * (tr['bbox'][3]-tr['bbox'][1])
            score = len(tr['times']) * 0.7 + area * 0.0005
            scores.append(score)
        main_id = active_tracks[np.argmax(scores)]['id']

    # ----------------------------
    # Hungarian matching
    # ----------------------------
    if len(active_tracks)>0 and len(detections)>0:
        cost_matrix = np.zeros((len(active_tracks), len(detections)))

        for i,track in enumerate(active_tracks):
            for j,(bbox,emb,_) in enumerate(detections):

                # embedding distance
                dist = min([cosine(emb,e) for e in track['emb_list']])

                # simple predicted center for short-term
                pred_center = track['center'] + (track['center'] - track['prev_center'])
                center_dist = np.linalg.norm(get_center(bbox)-pred_center)/200

                cost = 0.7*dist + 0.3*center_dist

                # Main character protection
                if track['id'] == main_id:
                    cost *= MAIN_CHAR_BONUS

                cost_matrix[i,j] = cost

        row,col = linear_sum_assignment(cost_matrix)

        matched_det = set()
        matched_track = set()

        # ----------------------------
        # Match update
        # ----------------------------
        for r,c in zip(row,col):
            if cost_matrix[r,c] < FEATURE_THRESHOLD:
                track = active_tracks[r]
                bbox,emb,gender = detections[c]

                # Prevent stealing the main character ID
                if main_id is not None and track['id'] == main_id:
                    dist_main = cosine(emb, track['embedding'])
                    if dist_main > REID_THRESHOLD:
                        continue

                update_embedding(track,emb)
                track['prev_center'] = track['center']
                track['bbox'] = bbox
                track['center'] = get_center(bbox)
                track['gender'] = gender
                track['missing'] = 0
                track['times'].append(t)

                matched_det.add(c)
                matched_track.add(r)

        # ----------------------------
        # unmatched tracks → lost
        # ----------------------------
        new_active = []
        for i,track in enumerate(active_tracks):
            if i not in matched_track:
                track['missing'] += 1
                if track['missing'] < MAX_MISSING:
                    lost_tracks.append(track)
            else:
                new_active.append(track)
        active_tracks = new_active

        # ----------------------------
        # unmatched detections → reconnect
        # ----------------------------
        for j,(bbox,emb,gender) in enumerate(detections):
            if j in matched_det:
                continue

            reconnected = False

            # 🔁 reconnect lost tracks with embedding + spatial tolerance
            for lt in lost_tracks:
                dist = min([cosine(emb,e) for e in lt['emb_list']])
                spatial_dist = np.linalg.norm(get_center(bbox)-lt['center'])
                if dist < REID_THRESHOLD and spatial_dist < SPATIAL_TOLERANCE:
                    lt['prev_center'] = lt['center']
                    lt['bbox'] = bbox
                    lt['center'] = get_center(bbox)
                    lt['missing'] = 0
                    lt['times'].append(t)
                    update_embedding(lt,emb)
                    active_tracks.append(lt)
                    lost_tracks.remove(lt)
                    reconnected = True
                    break

            # 🧠 identity bank
            if not reconnected:
                for person in identity_bank:
                    if person['id'] == main_id:
                        continue
                    dist = cosine(emb, person['embedding'])
                    if dist < REID_THRESHOLD:
                        new_t = create_track(emb,bbox,gender,t)
                        new_t['id'] = person['id']
                        active_tracks.append(new_t)
                        reconnected = True
                        break

            # 🆕 new ID
            if not reconnected:
                new_t = create_track(emb,bbox,gender,t)
                active_tracks.append(new_t)
                identity_bank.append({'id':new_t['id'],'embedding':emb})

    else:
        # first frame
        for bbox,emb,gender in detections:
            t_new = create_track(emb,bbox,gender,t)
            active_tracks.append(t_new)
            identity_bank.append({'id':t_new['id'],'embedding':emb})

    # ----------------------------
    # Draw
    # ----------------------------
    for tr in active_tracks:
        x1,y1,x2,y2 = map(int,tr['bbox'])
        color = (0,255,0)
        if tr['id'] == main_id:
            color = (0,0,255)
        label = f"ID{tr['id']}"
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,label,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

    cv2.imshow("Tracking",frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()