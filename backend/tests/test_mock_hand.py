import numpy as np
from types import SimpleNamespace

from ml.gesture_detector import GestureDetector


def create_mock_landmarks(finger_states, wrist=(0.5, 0.6, 0.0), scale=0.1):
    landmarks = np.zeros((21, 3), dtype=np.float32)
    wx, wy, wz = wrist
    landmarks[0] = [wx, wy, wz]  # Wrist
    
    # Base MCPs
    landmarks[5] = [wx - scale * 0.4, wy - scale * 1.0, wz]   # Index MCP
    landmarks[9] = [wx - scale * 0.1, wy - scale * 1.1, wz]   # Middle MCP
    landmarks[13] = [wx + scale * 0.2, wy - scale * 1.0, wz]  # Ring MCP
    landmarks[17] = [wx + scale * 0.5, wy - scale * 0.9, wz]  # Pinky MCP
    landmarks[1] = [wx - scale * 0.3, wy - scale * 0.3, wz]
    landmarks[2] = [wx - scale * 0.6, wy - scale * 0.6, wz]
    
    # Thumb
    if finger_states.get('thumb', False):
        if finger_states.get('thumb_up', False):
            landmarks[3] = [wx - scale * 0.2, wy - scale * 1.0, wz]
            landmarks[4] = [wx - scale * 0.2, wy - scale * 1.4, wz]  # up
        elif finger_states.get('thumb_down', False):
            landmarks[3] = [wx - scale * 0.2, wy + scale * 0.5, wz]
            landmarks[4] = [wx - scale * 0.2, wy + scale * 0.9, wz]  # down
        elif finger_states.get('thumb_touch_index', False):
            # Touching index tip
            idx_tip = [landmarks[5][0], landmarks[5][1] - scale * 1.3, wz]
            landmarks[4] = idx_tip
            landmarks[3] = [landmarks[2][0], landmarks[2][1] - scale * 0.5, wz]
        else:
            landmarks[3] = [wx - scale * 0.9, wy - scale * 0.9, wz]
            landmarks[4] = [wx - scale * 1.2, wy - scale * 1.1, wz]  # extended out
    else:
        landmarks[3] = [wx - scale * 0.3, wy - scale * 0.8, wz]
        landmarks[4] = [wx, wy - scale * 0.8, wz]  # folded across palm
        
    # Non-thumb fingers
    bases = [(5, 'index', -0.4), (9, 'middle', -0.1), (13, 'ring', 0.2), (17, 'pinky', 0.5)]
    for mcp_idx, name, x_off in bases:
        pip_idx = mcp_idx + 1
        dip_idx = mcp_idx + 2
        tip_idx = mcp_idx + 3
        mcp = landmarks[mcp_idx]
        
        if finger_states.get(name, False):
            # Extended straight up
            landmarks[pip_idx] = [mcp[0], mcp[1] - scale * 0.5, wz]
            landmarks[dip_idx] = [mcp[0], mcp[1] - scale * 0.9, wz]
            landmarks[tip_idx] = [mcp[0], mcp[1] - scale * 1.3, wz]
        else:
            # Curled into palm
            landmarks[pip_idx] = [mcp[0], mcp[1] - scale * 0.4, wz + scale * 0.2]
            landmarks[dip_idx] = [mcp[0], mcp[1] - scale * 0.1, wz + scale * 0.4]
            landmarks[tip_idx] = [mcp[0], mcp[1] + scale * 0.1, wz + scale * 0.2]
            
    # Wrap in MediaPipe-like structure
    proto_landmarks = [SimpleNamespace(x=pt[0], y=pt[1], z=pt[2]) for pt in landmarks]
    return SimpleNamespace(landmark=proto_landmarks)


def test_gestures():
    detector = GestureDetector()
    
    # 1. Test "I Love You" (Thumb, Index, Pinky extended, Middle & Ring folded)
    ily_hand = create_mock_landmarks({'thumb': True, 'index': True, 'middle': False, 'ring': False, 'pinky': True})
    label, conf, _ = detector.process_hand(ily_hand)
    print(f"ILY Test -> Label: {label}, Conf: {conf}")
    assert label == "I Love You", f"Expected 'I Love You', got {label}"
    
    # 2. Test "Peace" (Index and Middle extended, Ring, Pinky, Thumb folded)
    peace_hand = create_mock_landmarks({'thumb': False, 'index': True, 'middle': True, 'ring': False, 'pinky': False})
    label, conf, _ = detector.process_hand(peace_hand)
    print(f"Peace Test -> Label: {label}, Conf: {conf}")
    assert label == "Peace", f"Expected 'Peace', got {label}"

    # 3. Test "Thumbs Up"
    t_up = create_mock_landmarks({'thumb': True, 'thumb_up': True, 'index': False, 'middle': False, 'ring': False, 'pinky': False})
    label, conf, _ = detector.process_hand(t_up)
    print(f"Thumbs Up Test -> Label: {label}, Conf: {conf}")
    assert label == "Thumbs Up", f"Expected 'Thumbs Up', got {label}"

    # 4. Test "Wave / Hi"
    detector.reset()
    # Simulate a side-to-side waving sequence (open hand)
    open_hand_states = {'thumb': True, 'index': True, 'middle': True, 'ring': True, 'pinky': True}
    xs = [0.50, 0.53, 0.57, 0.54, 0.48, 0.44, 0.49, 0.55]
    final_label = None
    for i, x in enumerate(xs):
        h = create_mock_landmarks(open_hand_states, wrist=(x, 0.4, 0.0))
        lbl, conf, _ = detector.process_hand(h, t=10.0 + i * 0.1)
        if lbl:
            final_label = lbl
    print(f"Waving Test -> Final Label: {final_label}")
    assert final_label == "Hi", f"Expected 'Hi', got {final_label}"

    print("\nALL GESTURE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_gestures()
