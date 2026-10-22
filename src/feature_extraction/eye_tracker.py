"""
eye_tracker.py
--------------
Extracts real-time eye-tracking features from a webcam feed using
MediaPipe Face Mesh (468 landmarks, including iris landmarks).

Run this file directly on a machine WITH a webcam + display to sanity check
it on its own:  python3 eye_tracker.py

Features produced per frame (buffered, then aggregated into a window by
feature_aggregator.py):
  - EAR (eye aspect ratio)      -> drowsiness / blink proxy
  - blink events                -> counted to get blink rate
  - gaze point (normalized x,y) -> iris center relative to eye corners
  - pupil diameter (normalized) -> proxy for cognitive load
  - fixation vs saccade         -> derived from gaze-point velocity

Requires: opencv-python, mediapipe, numpy
    pip install opencv-python mediapipe numpy
"""

import time
import collections
import numpy as np

try:
    import cv2
    import mediapipe as mp
except ImportError as e:
    raise ImportError(
        "eye_tracker.py requires opencv-python and mediapipe. "
        "Install with: pip install opencv-python mediapipe"
    ) from e


# MediaPipe Face Mesh landmark indices for eyes / iris (refine_landmarks=True)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

# Landmarks used for head-pose (solvePnP): nose tip, chin, left/right eye
# outer corners, left/right mouth corners. Matched against a generic 3D
# face model below — not per-person calibrated, so treat yaw/pitch as
# "roughly this many degrees off-center", not a precise instrument.
_POSE_LANDMARK_IDX = {
    "nose_tip": 1, "chin": 152, "left_eye_outer": 263,
    "right_eye_outer": 33, "left_mouth": 291, "right_mouth": 61,
}
# generic 3D face model points (mm), in the same order as _POSE_LANDMARK_IDX
_MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),        # nose tip
    (0.0, -63.6, -12.5),    # chin
    (-43.3, 32.7, -26.0),   # left eye outer corner
    (43.3, 32.7, -26.0),    # right eye outer corner
    (-28.9, -28.9, -24.1),  # left mouth corner
    (28.9, -28.9, -24.1),   # right mouth corner
], dtype=np.float64)

EAR_BLINK_THRESHOLD = 0.21          # below this => eye considered closed
BLINK_CONSEC_FRAMES = 2             # min consecutive closed frames to count as a blink
SACCADE_VELOCITY_THRESHOLD = 0.06   # normalized gaze units / frame


def _euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def estimate_head_pose(landmarks, w, h):
    """
    Rough head yaw/pitch in degrees via solvePnP against a generic 3D face
    model (not per-person calibrated -- treat as approximate). Returns
    (yaw_deg, pitch_deg) or None if the pose can't be solved this frame.
    Yaw: negative = turned left, positive = turned right.
    Pitch: negative = tilted down, positive = tilted up.
    """
    try:
        image_points = np.array(
            [landmarks[_POSE_LANDMARK_IDX[k]] for k in
             ("nose_tip", "chin", "left_eye_outer", "right_eye_outer", "left_mouth", "right_mouth")],
            dtype=np.float64,
        )
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        ok, rvec, _tvec = cv2.solvePnP(
            _MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
        return float(yaw), float(pitch)
    except Exception:
        return None


def eye_aspect_ratio(landmarks, eye_idx):
    p = [landmarks[i] for i in eye_idx]
    # vertical distances
    v1 = _euclidean(p[1], p[5])
    v2 = _euclidean(p[2], p[4])
    # horizontal distance
    h = _euclidean(p[0], p[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


def iris_center(landmarks, iris_idx):
    pts = np.array([landmarks[i] for i in iris_idx])
    return pts.mean(axis=0)


def iris_diameter(landmarks, iris_idx):
    pts = np.array([landmarks[i] for i in iris_idx])
    # approximate diameter as max pairwise distance among iris ring points
    d = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = max(d, np.linalg.norm(pts[i] - pts[j]))
    return d


class EyeTracker:
    """
    Wraps MediaPipe FaceMesh and produces a stream of raw per-frame
    measurements. Call `read_frame_features()` in a loop; feed the results
    into FeatureAggregator (feature_aggregator.py) to get windowed features
    for the model.
    """

    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera_index / permissions.")

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,     # needed for iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._closed_frame_count = 0
        self._prev_gaze = None
        self._prev_time = None

    def _to_pixel_landmarks(self, face_landmarks, w, h):
        return [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]

    def read_frame_features(self):
        """
        Returns a dict of raw instantaneous measurements for one frame,
        or None if no face was detected:
          {
            "timestamp": float,
            "ear": float,
            "is_blink": bool,          # True on the frame a blink is confirmed
            "gaze_x": float, "gaze_y": float,  # normalized [0,1] gaze point
            "pupil_diameter": float,   # normalized by eye width
            "is_saccade": bool,
            "head_pose": (yaw_deg, pitch_deg) or None,  # rough head orientation
          }
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        now = time.time()
        if not results.multi_face_landmarks:
            return None

        landmarks = self._to_pixel_landmarks(results.multi_face_landmarks[0], w, h)

        ear_left = eye_aspect_ratio(landmarks, LEFT_EYE)
        ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (ear_left + ear_right) / 2.0

        # blink detection (consecutive low-EAR frames)
        is_blink = False
        if ear < EAR_BLINK_THRESHOLD:
            self._closed_frame_count += 1
        else:
            if self._closed_frame_count >= BLINK_CONSEC_FRAMES:
                is_blink = True
            self._closed_frame_count = 0

        # gaze point: iris center relative to eye bounding box, normalized
        left_iris_c = iris_center(landmarks, LEFT_IRIS)
        right_iris_c = iris_center(landmarks, RIGHT_IRIS)
        gaze_pt = (left_iris_c + right_iris_c) / 2.0
        gaze_x, gaze_y = gaze_pt[0] / w, gaze_pt[1] / h

        # pupil diameter normalized by eye width (so it's scale-invariant to
        # distance from camera)
        eye_width = _euclidean(landmarks[LEFT_EYE[0]], landmarks[LEFT_EYE[3]])
        pupil_d_left = iris_diameter(landmarks, LEFT_IRIS)
        pupil_d_right = iris_diameter(landmarks, RIGHT_IRIS)
        pupil_norm = ((pupil_d_left + pupil_d_right) / 2.0) / (eye_width + 1e-6)

        # saccade detection via gaze velocity between frames
        is_saccade = False
        if self._prev_gaze is not None and self._prev_time is not None:
            dt = max(now - self._prev_time, 1e-3)
            vel = np.linalg.norm(np.array([gaze_x, gaze_y]) - np.array(self._prev_gaze)) / dt
            is_saccade = vel > SACCADE_VELOCITY_THRESHOLD
        self._prev_gaze = (gaze_x, gaze_y)
        self._prev_time = now

        return {
            "timestamp": now,
            "ear": ear,
            "is_blink": is_blink,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "pupil_diameter": pupil_norm,
            "is_saccade": is_saccade,
            "head_pose": estimate_head_pose(landmarks, w, h),  # (yaw_deg, pitch_deg) or None
        }

    def release(self):
        self.cap.release()


if __name__ == "__main__":
    # Standalone debug run: prints live EAR/gaze/pupil measurements.
    # Press Ctrl+C to quit.
    tracker = EyeTracker()
    print("Reading webcam... press Ctrl+C to quit.")
    try:
        while True:
            feats = tracker.read_frame_features()
            if feats:
                yaw_pitch = feats["head_pose"]
                pose_str = f"yaw={yaw_pitch[0]:.1f} pitch={yaw_pitch[1]:.1f}" if yaw_pitch else "pose=?"
                print(
                    f"ear={feats['ear']:.3f}  blink={feats['is_blink']}  "
                    f"gaze=({feats['gaze_x']:.3f},{feats['gaze_y']:.3f})  "
                    f"pupil={feats['pupil_diameter']:.3f}  saccade={feats['is_saccade']}  {pose_str}"
                )
    except KeyboardInterrupt:
        pass
    finally:
        tracker.release()
