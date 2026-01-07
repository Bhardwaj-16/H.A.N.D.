import cv2
import mediapipe as mp

from packages.hand_landmarks import get_frame_and_landmarks
from packages.gestures import detect_gestures, detect_zoom

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

prev_zoom_dist = None
baseline_zoom_dist = None

while cap.isOpened():
    frame, result = get_frame_and_landmarks(cap)
    if frame is None:
        break

    gesture = "NONE"
    finger_positions = None

    if result.multi_hand_landmarks:
        if len(result.multi_hand_landmarks) == 2:
            gesture, prev_zoom_dist, baseline_zoom_dist, finger_positions = detect_zoom(
                result.multi_hand_landmarks[0],
                result.multi_hand_landmarks[1],
                prev_zoom_dist,
                baseline_zoom_dist
            )

        else:
            prev_zoom_dist = None
            baseline_zoom_dist = None
            for hand_landmarks in result.multi_hand_landmarks:
                gesture = detect_gestures(hand_landmarks)

        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS
            )
    
    else:
        prev_zoom_dist = None
        baseline_zoom_dist = None

    if finger_positions is not None:
        h, w, _ = frame.shape
        for finger_pos in finger_positions:
            x, y = finger_pos
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 10, (0, 165, 255), -1)  # Orange dot

    cv2.putText(
        frame,
        gesture,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("H.A.N.D.", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()