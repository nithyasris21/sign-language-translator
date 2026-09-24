"""MediaPipe Holistic wrapper: video frame -> flat landmark feature vector."""
import base64

import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def make_holistic(static: bool = False):
    return mp_holistic.Holistic(
        static_image_mode=static,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def detect(frame_bgr, holistic):
    """Run Holistic on a BGR frame. Returns the raw MediaPipe results."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)
    return results


LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


def _body_frame(pose_xyz):
    """Origin and scale of the signer's body, from the shoulder line.

    MediaPipe returns coordinates normalised to the *image*, so a signer
    standing left-of-centre or filmed from further back produces completely
    different numbers for the same sign. Re-expressing every point relative to
    the shoulders removes the framing and leaves the gesture.
    """
    left, right = pose_xyz[LEFT_SHOULDER], pose_xyz[RIGHT_SHOULDER]
    origin = (left + right) / 2.0
    scale = float(np.linalg.norm(left[:2] - right[:2]))
    if scale < 1e-3:          # shoulders missing or degenerate; don't divide by ~0
        scale = 1.0
    return origin, scale


def extract_keypoints(results) -> np.ndarray:
    """Flatten pose + both hands into a fixed-size vector (258,).

    Face mesh is deliberately excluded: 468 extra points dominate the vector
    and most lexical signs are carried by hands + upper body.

    Everything is expressed in the signer's own body frame (see _body_frame),
    so the vector describes the gesture rather than the camera setup. Blocks
    for undetected hands stay all-zero, which is what the model's Masking
    layer and preprocess.hand_fraction both key off.
    """
    has_pose = results.pose_landmarks is not None
    pose_xyz = (
        np.array([[l.x, l.y, l.z] for l in results.pose_landmarks.landmark], dtype=np.float32)
        if has_pose
        else np.zeros((33, 3), dtype=np.float32)
    )
    visibility = (
        np.array([[l.visibility] for l in results.pose_landmarks.landmark], dtype=np.float32)
        if has_pose
        else np.zeros((33, 1), dtype=np.float32)
    )

    def hand(landmarks):
        if landmarks is None:
            return np.zeros((21, 3), dtype=np.float32), False
        return np.array([[l.x, l.y, l.z] for l in landmarks.landmark], dtype=np.float32), True

    lh_xyz, lh_seen = hand(results.left_hand_landmarks)
    rh_xyz, rh_seen = hand(results.right_hand_landmarks)

    if has_pose:
        origin, scale = _body_frame(pose_xyz)
        pose_xyz = (pose_xyz - origin) / scale
        if lh_seen:
            lh_xyz = (lh_xyz - origin) / scale
        if rh_seen:
            rh_xyz = (rh_xyz - origin) / scale

    pose = np.concatenate([pose_xyz, visibility], axis=1).flatten()
    return np.concatenate([pose, lh_xyz.flatten(), rh_xyz.flatten()]).astype(np.float32)


def has_hands(results) -> bool:
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None


def draw(frame_bgr, results):
    """Overlay skeleton on the frame (used by the data-collection script)."""
    mp_drawing.draw_landmarks(frame_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return frame_bgr


def decode_data_url(data_url: str):
    """'data:image/jpeg;base64,...' (or bare base64) -> BGR ndarray, or None
    if it is not a decodable image. One bad frame must not end the stream."""
    payload = data_url.split(",", 1)[-1]
    try:
        buf = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    except (ValueError, cv2.error):  # binascii.Error is a ValueError
        return None
