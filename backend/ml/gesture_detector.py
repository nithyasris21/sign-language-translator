"""Hand gesture and sign recognition engine.

Detects instant hand signs and gestures from MediaPipe landmarks:
- "Hi" / "Hello" (waving hand or open palm greeting)
- "I Love You" (ASL ILY: thumb + index + pinky extended)
- "Peace" / "Victory" (V sign: index + middle extended)
- "Thumbs Up" / "Yes" / "Good"
- "Thumbs Down" / "No" / "Bad"
- "OK" (index + thumb circle, 3 fingers extended)
- "Stop" / "Wait" (steady open palm)
- "Thank You" (flat hand near chin moving forward)
- "Rock On" (horns: index + pinky extended)
- "Call Me" (shaka: thumb + pinky extended)
- ASL Alphabet letters: 'A', 'B', 'C', 'D', 'L', 'V', 'W', 'Y'
"""
import time
from collections import deque
import numpy as np


class GestureDetector:
    def __init__(self, history_len: int = 24):
        self.history_len = history_len
        # History tracks (timestamp, wrist_x, wrist_y, hand_side, fingers_extended)
        self.history = deque(maxlen=history_len)
        self._last_detected = None
        self._detect_count = 0
        self._min_consecutive_frames = 2  # require 2 consecutive frames for static gestures to avoid noise

    def reset(self):
        self.history.clear()
        self._last_detected = None
        self._detect_count = 0

    @staticmethod
    def _finger_states(landmarks: np.ndarray) -> dict[str, bool]:
        """Determine extension state of all 5 fingers.
        
        landmarks: shape (21, 3), coords [x, y, z]
        """
        wrist = landmarks[0]
        # Palm scale is distance between index MCP (5) and pinky MCP (17)
        palm_scale = float(np.linalg.norm(landmarks[5][:2] - landmarks[17][:2]))
        if palm_scale < 1e-4:
            palm_scale = 0.1

        # Non-thumb fingers: Index (5..8), Middle (9..12), Ring (13..16), Pinky (17..20)
        finger_indices = {
            "index": (5, 6, 7, 8),
            "middle": (9, 10, 11, 12),
            "ring": (13, 14, 15, 16),
            "pinky": (17, 18, 19, 20),
        }
        
        states = {}
        for name, (mcp, pip, dip, tip) in finger_indices.items():
            d_wrist_tip = float(np.linalg.norm(landmarks[tip] - wrist))
            d_wrist_pip = float(np.linalg.norm(landmarks[pip] - wrist))
            d_mcp_tip = float(np.linalg.norm(landmarks[tip] - landmarks[mcp]))
            d_mcp_pip = float(np.linalg.norm(landmarks[pip] - landmarks[mcp]))
            
            # Tip is further from wrist than PIP and finger is uncurled
            is_extended = (d_wrist_tip > d_wrist_pip * 1.1) and (d_mcp_tip > d_mcp_pip * 1.3)
            # Also check if tip is below MCP in inverted coords (upright hand)
            if landmarks[tip][1] < landmarks[pip][1] < landmarks[mcp][1]:
                is_extended = True
            elif landmarks[tip][1] > landmarks[pip][1] and d_wrist_tip < d_wrist_pip:
                is_extended = False
                
            states[name] = is_extended

        # Thumb (1: CMC, 2: MCP, 3: IP, 4: TIP)
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]

        d_thumb_pinky = float(np.linalg.norm(thumb_tip - pinky_mcp))
        d_thumb_index = float(np.linalg.norm(thumb_tip - index_mcp))
        
        # Extended thumb points away from palm
        thumb_extended = (d_thumb_pinky > 1.2 * palm_scale) and (d_thumb_index > 0.5 * palm_scale)
        states["thumb"] = thumb_extended
        
        # Additional geometric properties
        states["palm_scale"] = palm_scale
        states["thumb_up"] = (thumb_tip[1] < thumb_mcp[1] - 0.3 * palm_scale) and thumb_extended
        states["thumb_down"] = (thumb_tip[1] > thumb_mcp[1] + 0.3 * palm_scale) and thumb_extended
        states["thumb_index_touch"] = float(np.linalg.norm(thumb_tip - landmarks[8])) < (0.45 * palm_scale)
        
        return states

    def _detect_wave(self, hand_side: str) -> bool:
        """Detect side-to-side waving oscillation in recent frames."""
        pts = [h for h in self.history if h["side"] == hand_side and h["fingers_open"] >= 3]
        if len(pts) < 6:
            return False
        
        xs = [p["x"] for p in pts]
        span = max(xs) - min(xs)
        if span < 0.03:  # minimum amplitude of movement
            return False

        # Count direction changes
        dx = [xs[i] - xs[i-1] for i in range(1, len(xs))]
        direction_changes = 0
        current_dir = 0  # 1 for right, -1 for left
        for d in dx:
            if abs(d) > 0.004:
                new_dir = 1 if d > 0 else -1
                if current_dir != 0 and new_dir != current_dir:
                    direction_changes += 1
                current_dir = new_dir

        return direction_changes >= 2

    def process_hand(self, landmarks_raw, hand_side: str = "right", pose_raw = None, t: float = None) -> tuple[str | None, float, dict]:
        """Analyze a single hand landmarks object from MediaPipe.
        
        Returns: (label, confidence, scores_dict) or (None, 0.0, {})
        """
        if landmarks_raw is None:
            return None, 0.0, {}

        t = time.monotonic() if t is None else t
        landmarks = np.array([[l.x, l.y, l.z] for l in landmarks_raw.landmark], dtype=np.float32)
        wrist = landmarks[0]
        
        f = self._finger_states(landmarks)
        t_ext = f["thumb"]
        i_ext = f["index"]
        m_ext = f["middle"]
        r_ext = f["ring"]
        p_ext = f["pinky"]
        num_open = sum([i_ext, m_ext, r_ext, p_ext])

        self.history.append({
            "t": t,
            "x": float(wrist[0]),
            "y": float(wrist[1]),
            "side": hand_side,
            "fingers_open": num_open + (1 if t_ext else 0),
        })

        is_waving = self._detect_wave(hand_side)

        scores: dict[str, float] = {}

        # 1. WAVE / "Hi" / "Hello"
        # Raised hand with open fingers waving, or held high in greeting
        if (num_open >= 3) and (wrist[1] < 0.85):
            if is_waving:
                scores["Hi"] = 0.98
            elif wrist[1] < 0.55 and i_ext and m_ext and r_ext and p_ext:
                # Open hand held high (greeting / ASL hello pose)
                scores["Hi"] = 0.92
            elif num_open == 4 and t_ext:
                # Open palm facing camera
                scores["Hi"] = 0.85
                scores["Stop"] = 0.80

        # 2. I LOVE YOU (🤟: Thumb, Index, Pinky extended, Middle & Ring folded)
        if t_ext and i_ext and p_ext and not m_ext and not r_ext:
            scores["I Love You"] = 0.98

        # 3. PEACE / VICTORY (✌️: Index and Middle extended in V, Ring, Pinky, Thumb closed)
        if i_ext and m_ext and not r_ext and not p_ext:
            # Check separation between tips
            d_im = float(np.linalg.norm(landmarks[8] - landmarks[12]))
            if d_im > 0.25 * f["palm_scale"]:
                scores["Peace"] = 0.97
            else:
                scores["Peace"] = 0.90

        # 4. THUMBS UP (👍: Thumb pointing up, all other 4 fingers folded)
        if f["thumb_up"] and (num_open == 0 or (num_open == 1 and not i_ext and not m_ext)):
            scores["Thumbs Up"] = 0.97
            scores["Yes"] = 0.90

        # 5. THUMBS DOWN (👎: Thumb pointing down, all other 4 fingers folded)
        if f["thumb_down"] and num_open == 0:
            scores["Thumbs Down"] = 0.97
            scores["No"] = 0.90

        # 6. OK (👌: Thumb and Index tips touch, other 3 fingers extended)
        if f["thumb_index_touch"] and m_ext and (r_ext or p_ext):
            scores["OK"] = 0.96

        # 7. ROCK ON (🤘: Index and Pinky extended, Middle, Ring, Thumb closed)
        if i_ext and p_ext and not m_ext and not r_ext and not t_ext:
            scores["Rock On"] = 0.96

        # 8. CALL ME (🤙: Thumb and Pinky extended, Index, Middle, Ring closed)
        if t_ext and p_ext and not i_ext and not m_ext and not r_ext:
            scores["Call Me"] = 0.96

        # 9. STOP / WAIT (✋: Flat open hand steady, fingers upright)
        if num_open == 4 and t_ext and not is_waving and wrist[1] < 0.75:
            scores["Stop"] = max(scores.get("Stop", 0.0), 0.92)

        # 10. ASL Alphabet Letters:
        # 'L': Index up, Thumb out at 90 deg, others closed
        if i_ext and t_ext and not m_ext and not r_ext and not p_ext:
            scores["L"] = 0.95
        # 'W': Index, Middle, Ring up, Pinky closed
        if i_ext and m_ext and r_ext and not p_ext:
            scores["W"] = 0.95
        # 'Y': Thumb and Pinky extended (like Call Me / Shaka)
        if t_ext and p_ext and not i_ext and not m_ext and not r_ext:
            scores["Y"] = 0.93

        # 11. THANK YOU (flat hand near chin / face moving forward)
        if pose_raw is not None and num_open >= 3 and not is_waving:
            # Nose landmark is 0 in pose
            nose = pose_raw.landmark[0] if len(pose_raw.landmark) > 0 else None
            if nose and abs(wrist[1] - nose.y) < 0.35 and abs(wrist[0] - nose.x) < 0.35:
                scores["Thank You"] = 0.88

        if not scores:
            return None, 0.0, {}

        # Pick top scoring label
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]

        # Require minimum consecutive detection for static labels to prevent flicker
        if best_label == self._last_detected:
            self._detect_count += 1
        else:
            self._last_detected = best_label
            self._detect_count = 1

        # Waving or high-confidence gesture fires promptly
        if is_waving or self._detect_count >= self._min_consecutive_frames or best_score >= 0.95:
            return best_label, best_score, scores

        return None, 0.0, scores


gesture_engine = GestureDetector()
