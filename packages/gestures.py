def is_finger_extended(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def detect_gesture(hand_landmarks):
    lm = hand_landmarks.landmark

    index_up = is_finger_extended(lm, 8, 6)
    middle_up = is_finger_extended(lm, 12, 10)
    ring_up = is_finger_extended(lm, 16, 14)
    pinky_up = is_finger_extended(lm, 20, 18)

    if not (index_up or middle_up or ring_up or pinky_up):
        return "DRAG"
    
    if index_up and not (middle_up or ring_up or pinky_up):
        return "ROTATE"
    
    return "NONE"